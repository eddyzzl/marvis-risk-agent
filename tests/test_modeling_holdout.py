"""Sealed-mode software acceptance on real files, native tools and local HTTP."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
from io import BytesIO
import json
import sqlite3
import threading

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from marvis.app import create_app
from marvis.data.backend import DataBackend
from marvis.data.registry import DatasetRegistry
from marvis.db_schema import connect
from marvis.modeling_holdout.contracts import HoldoutError
from marvis.modeling_holdout.service import HoldoutService
from marvis.plugins.manifest import ToolRef
from marvis.plugins.subprocess_worker import _assert_file_access
from marvis.repositories.datasets import DatasetRepository


def _bytes(frame):
    output = BytesIO()
    frame.to_parquet(output, index=False)
    return output.getvalue()


def _frame(kind='binary', n=240, seed=53):
    rng = np.random.default_rng(seed)
    target = np.arange(n) % (3 if kind == 'multiclass' else 2)
    if kind == 'continuous':
        target = rng.normal(size=n)
    return pd.DataFrame({'x': target * 0.6 + rng.normal(size=n),
                         'z': rng.normal(size=n), 'y': target,
                         'split': ['train'] * (n * 2 // 3) + ['test'] * (n - n * 2 // 3)})


@pytest.fixture
def custody(tmp_path):
    app = create_app(tmp_path / 'workspace')
    with ExitStack() as stack:
        clients, ids = {}, {}
        for name, role in (('custodian', 'admin'), ('developer', 'maker'), ('evaluator', 'checker'), ('other', 'checker')):
            client = TestClient(app)
            if name == 'custodian':
                stack.enter_context(client)
            else:
                stack.callback(client.close)
            response = client.post('/api/production-governance/principals/claim',
                headers={'x-marvis-governance-admin': app.state.plugin_admin_token},
                json={'display_name': name, 'role': role})
            assert response.status_code == 201, response.text
            clients[name], ids[name] = client, response.json()['id']
        source = app.state.settings.workspace / 'empty-source'
        source.mkdir()
        response = clients['custodian'].post('/api/tasks', json={
            'model_name': 'Synthetic blind mechanism', 'validator': 'qa',
            'source_dir': str(source), 'task_type': 'modeling', 'run_mode': 'manual', 'recipes': ['lr'],
        })
        assert response.status_code == 200, response.text
        task = response.json()['id']
        yield app, clients, ids, task


def _seal(custody, kind='binary'):
    app, clients, ids, task = custody
    holdout = _frame(kind, n=90, seed=181).drop(columns='split')
    payload = {'data_identity': 'synthetic-independent-cohort', 'developer_id': ids['developer'],
               'evaluator_id': ids['evaluator'], 'target_col': 'y', 'split_col': 'split',
               'target_type': kind, 'authority_basis': 'Authorized synthetic fixture for software mechanism only'}
    raw = _bytes(holdout)
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',
        files={'file': ('independent.parquet', raw, 'application/octet-stream')},
        data={'contract': json.dumps(payload)})
    assert response.status_code == 201, response.text
    return raw, payload, response.json()


def _develop(custody, kind='binary', recipe='lr', params=None):
    app, clients, _, task = custody
    frame = _frame(kind)
    response = clients['developer'].post(f'/api/tasks/{task}/blind-oot/development',
        files={'file': ('development.parquet', _bytes(frame), 'application/octet-stream')})
    assert response.status_code == 201, response.text
    inputs = {'dataset_id': response.json()['dataset_id'], 'recipe': recipe, 'features': ['x', 'z'],
              'target_col': 'y', 'split_col': 'split', 'split_values': {'train': 'train', 'test': 'test'},
              'seed': 19, 'params': params or {}, 'target_type': kind}
    result = app.state.tool_runner.invoke(ToolRef('modeling', 'train_model'), inputs, task_id=task)
    assert result.ok, result.error
    assert result.output['metrics']['oot_ks'] is None
    assert result.output['metrics']['oot_auc'] is None
    return inputs, result.output


def _freeze(custody, output):
    _, clients, _, task = custody
    response = clients['developer'].post(f'/api/tasks/{task}/blind-oot/freeze',
                                         json={'artifact_id': output['artifact_id']})
    assert response.status_code == 200, response.text
    return response.json()['freeze_hash']


@pytest.mark.parametrize(('kind', 'recipe', 'metric'), [('binary', 'lr', 'auc'),
    ('continuous', 'lr_regressor', 'rmse'), ('multiclass', 'lr_multiclass', 'macro_auc')])
def test_real_blind_api_freezes_and_evaluates_once_without_agent_oot(custody, monkeypatch, kind, recipe, metric):
    app, clients, _, task = custody
    reads = []
    original = HoldoutService._score_sealed

    def observe(self, cohort, closure, scorer):
        with connect(self.db_path) as conn:
            ledger = conn.execute('SELECT * FROM blind_oot_evaluations WHERE cohort_id=?', (cohort['id'],)).fetchone()
            assert ledger['state'] == 'reading'
            assert ledger['read_started_at'] is not None
        reads.append(cohort['id'])
        return original(self, cohort, closure, scorer)

    monkeypatch.setattr(HoldoutService, '_score_sealed', observe)
    raw, _, seal = _seal(custody, kind)
    assert not reads
    assert clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': '0' * 64}).status_code == 409
    assert not reads
    inputs, output = _develop(custody, kind, recipe)
    assert not reads
    frozen = _freeze(custody, output)
    assert not reads
    for name in ('developer', 'custodian', 'other'):
        assert clients[name].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen}).status_code == 403
    result = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen, 'request_id': 'first'})
    assert result.status_code == 200, result.text
    assert np.isfinite(result.json()['metrics'][metric])
    assert result.json()['metrics']['rows'] == 90
    repeat = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen, 'request_id': 'changed-id'})
    assert repeat.status_code == 200 and repeat.json() == result.json()
    assert reads == [seal['cohort_id']]
    assert result.json()['historical_availability'] == 'unknown'
    assert result.json()['label_maturity'] == 'unknown'
    assert clients['developer'].get(f'/api/tasks/{task}/blind-oot/result').status_code == 403
    public = clients['developer'].get(f'/api/tasks/{task}/experiments')
    assert public.status_code == 200, public.text
    assert public.json()['experiments'][0]['metrics']['oot_auc'] is None
    assert 'sealed_hash' not in public.text
    assert clients['evaluator'].get(f'/api/tasks/{task}/agent/messages').status_code == 403
    assert clients['developer'].delete(f'/api/tasks/{task}').status_code == 409
    blocked = app.state.tool_runner.invoke(ToolRef('modeling', 'train_model'), inputs, task_id=task)
    assert not blocked.ok and 'blind_development_is_frozen' in blocked.error
    private = app.state.settings.workspace / '.blind_oot'
    originals = list(private.glob('sealed/*/original.parquet'))
    assert len(originals) == 1 and originals[0].read_bytes() == raw
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM blind_oot_evaluations').fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute('DELETE FROM blind_oot_evaluations')


def test_sealed_original_denied_to_registry_backend_generic_plugin_and_development(custody, tmp_path, monkeypatch):
    app, clients, _, task = custody
    raw, _, _ = _seal(custody)
    original = next((app.state.settings.workspace / '.blind_oot').glob('sealed/*/original.parquet'))
    backend = DataBackend(app.state.settings.datasets_dir)
    registry = DatasetRegistry(DatasetRepository(app.state.settings.db_path), backend, app.state.settings.datasets_dir)
    copied = tmp_path / 'renamed.parquet'
    copied.write_bytes(raw)
    for path in (original, copied):
        with pytest.raises(HoldoutError, match='sealed_holdout_access_forbidden'):
            backend.read_frame(path)
        with pytest.raises(HoldoutError, match='sealed_holdout_access_forbidden'):
            registry.register_from_upload(task, path)
        with pytest.raises(HoldoutError, match='sealed_holdout_access_forbidden'):
            registry.register_existing(path, task_id=task, role='sample')
    with pytest.raises(PermissionError, match='sealed_holdout_access_forbidden'):
        _assert_file_access(original, 'read', (app.state.settings.workspace,), ())
    response = clients['developer'].post(f'/api/tasks/{task}/blind-oot/development',
        files={'file': ('renamed.parquet', raw, 'application/octet-stream')})
    assert response.status_code == 403, response.text
    module = tmp_path / 'escape.py'
    module.write_text("def run(inputs,ctx):\n return {'secret': open(inputs['path']).read()}\n")
    result = app.state.tool_runner.invoke_adhoc(module=module, entrypoint='run', inputs={'path': str(original)},
        input_schema={'type': 'object'}, output_schema={'type': 'object'}, timeout_seconds=10, task_id=task)
    assert not result.ok and result.error == 'blind_mode_arbitrary_execution_forbidden'


def test_legacy_exposed_source_cannot_be_reclassified_blind(custody, tmp_path):
    app, clients, ids, task = custody
    raw = _bytes(_frame(n=90, seed=181).drop(columns='split'))
    path = tmp_path / 'previously-exposed.parquet'
    path.write_bytes(raw)
    backend = DataBackend(app.state.settings.datasets_dir)
    registry = DatasetRegistry(DatasetRepository(app.state.settings.db_path), backend, app.state.settings.datasets_dir)
    registry.register_from_upload(task, path)
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',
        files={'file': ('original.parquet', raw, 'application/octet-stream')},
        data={'contract': json.dumps({'data_identity': 'new-label', 'developer_id': ids['developer'],
             'evaluator_id': ids['evaluator'], 'target_col': 'y', 'split_col': 'split', 'authority_basis': 'synthetic'})})
    assert response.status_code == 409, response.text
    assert response.json()['detail']['code'] == 'blind_intake_must_precede_development'


def test_concurrent_evaluation_reserves_one_cohort_before_read(custody, monkeypatch):
    _, clients, _, task = custody
    _seal(custody)
    _, output = _develop(custody)
    frozen = _freeze(custody, output)
    entered, release = threading.Event(), threading.Event()
    reads = []
    original = HoldoutService._score_sealed

    def blocked(self, cohort, closure, scorer):
        reads.append(cohort['id'])
        entered.set()
        assert release.wait(15)
        return original(self, cohort, closure, scorer)

    monkeypatch.setattr(HoldoutService, '_score_sealed', blocked)
    client = clients['evaluator']
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(client.post, f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen, 'request_id': 'one'})
        assert entered.wait(15)
        second = client.post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen, 'request_id': 'two'})
        assert second.status_code == 409, second.text
        release.set()
        completed = first.result(timeout=15)
        assert completed.status_code == 200, completed.text
    assert len(reads) == 1


def test_failure_after_read_cannot_reopen_budget_and_signed_result_recovery_does_not_rescore(custody, monkeypatch):
    app, clients, _, task = custody
    _seal(custody)
    _, output = _develop(custody)
    frozen = _freeze(custody, output)
    reads = []
    original = HoldoutService._score_sealed

    def observe(self, cohort, closure, scorer):
        reads.append(cohort['id'])
        return original(self, cohort, closure, scorer)

    def lost_commit(self, cohort, owner, result_hash):
        raise OSError('synthetic lost ledger commit')

    monkeypatch.setattr(HoldoutService, '_score_sealed', observe)
    monkeypatch.setattr(HoldoutService, '_commit_result', lost_commit)
    failed = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen})
    assert failed.status_code == 409, failed.text
    assert clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen, 'request_id': 'reset'}).status_code == 409
    assert clients['custodian'].post(f'/api/tasks/{task}/blind-oot/recoveries',
        json={'mode': 'resume_before_read', 'reason': 'Synthetic recovery test'}).status_code == 409
    recovery = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/recoveries',
        json={'mode': 'publish_result', 'reason': 'Publish already signed result without any new sample read'})
    assert recovery.status_code == 201, recovery.text
    repaired = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate',
        json={'freeze_hash': frozen, 'recovery_id': recovery.json()['recovery_id']})
    assert repaired.status_code == 200, repaired.text
    assert len(reads) == 1
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT state FROM blind_oot_evaluations').fetchone()[0] == 'committed'


def test_freeze_pins_model_bytes_and_rejects_a_second_candidate(custody):
    app, clients, _, task = custody
    _seal(custody)
    inputs, first = _develop(custody)
    second = app.state.tool_runner.invoke(ToolRef('modeling', 'train_model'), inputs, task_id=task)
    assert second.ok, second.error
    frozen = _freeze(custody, first)
    changed = clients['developer'].post(f'/api/tasks/{task}/blind-oot/freeze', json={'artifact_id': second.output['artifact_id']})
    assert changed.status_code == 409, changed.text
    artifact = app.state.settings.tasks_dir / task / 'modeling_artifacts'
    for path in artifact.glob('*.joblib'):
        path.write_bytes(b'changed after final freeze')
    result = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen})
    assert result.status_code == 200, result.text
    closure = app.state.settings.workspace / '.blind_oot' / 'freezes' / frozen / 'closure.json'
    assert hashlib.sha256(closure.read_bytes()).hexdigest() == frozen


@pytest.mark.parametrize(('kind', 'recipe', 'params'), [
    ('binary', 'lgb', {'n_estimators': 12}), ('binary', 'xgb', {'n_estimators': 12}),
    ('binary', 'catboost', {'iterations': 12}), ('binary', 'mlp', {'max_iter': 80, 'hidden_layer_sizes': [6]}),
    ('binary', 'scorecard', {}), ('binary', 'ensemble', {'base_recipe': 'lr', 'n_members': 2}),
    ('continuous', 'lgb_regressor', {'n_estimators': 12}), ('continuous', 'xgb_regressor', {'n_estimators': 12}),
    ('continuous', 'mlp_regressor', {'max_iter': 80, 'hidden_layer_sizes': [6]}),
    ('multiclass', 'lgb_multiclass', {'n_estimators': 12}), ('multiclass', 'xgb_multiclass', {'n_estimators': 12}),
    ('multiclass', 'mlp_multiclass', {'max_iter': 80, 'hidden_layer_sizes': [6]}),
])
def test_all_declared_native_profiles_have_reachable_freeze_and_score(custody, kind, recipe, params):
    _, clients, _, task = custody
    _seal(custody, kind)
    _, output = _develop(custody, kind, recipe, params)
    frozen = _freeze(custody, output)
    response = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen})
    assert response.status_code == 200, response.text
    assert all(np.isfinite(value) for value in response.json()['metrics'].values())


def test_summary_and_actions_are_bound_and_caller_json_cannot_forge_native_freeze(custody):
    _, clients, ids, task = custody
    _, payload, _ = _seal(custody)
    for name in ('custodian', 'developer', 'evaluator'):
        task_response = clients[name].get(f'/api/tasks/{task}')
        assert task_response.status_code == 200
        assert task_response.json()['blind_oot']['viewer_role'] == {'custodian':'custody','developer':'develop','evaluator':'evaluate'}[name]
        assert task in {item['id'] for item in clients[name].get('/api/tasks').json()}
        response = clients[name].get(f'/api/tasks/{task}/blind-oot')
        assert response.status_code == 200 and response.json()['viewer_role'] == {'custodian':'custody','developer':'develop','evaluator':'evaluate'}[name]
        assert response.json()['freeze_hash'] is None
    assert clients['other'].get(f'/api/tasks/{task}').status_code == 403
    assert task not in {item['id'] for item in clients['other'].get('/api/tasks').json()}
    response = clients['custodian'].get(f'/api/tasks/{task}/blind-oot/participants')
    assert response.status_code == 200
    assert ids['developer'] in {x['id'] for x in response.json()['participants']}
    _, output = _develop(custody)
    candidates = clients['developer'].get(f'/api/tasks/{task}/blind-oot/candidates').json()['candidates']
    assert [x['artifact_id'] for x in candidates] == [output['artifact_id']]
    response = clients['developer'].post(f'/api/tasks/{task}/blind-oot/freeze',
        json={'artifact_id': output['artifact_id'], 'native_receipt_verified': True})
    assert response.status_code == 422
    _freeze(custody, output)
    summary = clients['developer'].get(f'/api/tasks/{task}/blind-oot').json()
    assert summary['model_evidence']['native_receipt_verified'] is True
    assert 'sealed_hash' not in summary and 'authority_basis' not in summary
    assert payload['target_col'] == 'y'


def test_pre_read_failure_safe_approval_resumes_without_budget_reset(custody, monkeypatch):
    app, clients, _, task = custody
    _seal(custody)
    _, output = _develop(custody)
    frozen = _freeze(custody, output)
    original = HoldoutService._mark_read_started
    def fail(*args):
        raise OSError('synthetic pre-read carrier failure')
    monkeypatch.setattr(HoldoutService, '_mark_read_started', fail)
    url = f'/api/tasks/{task}/blind-oot'
    failed = clients['evaluator'].post(url + '/evaluate', json={'freeze_hash': frozen})
    assert failed.status_code == 409 and 'synthetic pre-read' not in failed.text
    summary = clients['custodian'].get(url).json()
    assert summary['state'] == 'uncertain' and not summary['sample_read_started']
    assert summary['recovery_options'] == ['resume_before_read']
    approved = clients['custodian'].post(url + '/recoveries', json={'mode': 'resume_before_read', 'reason': 'Carrier repaired before any sample read'})
    assert approved.status_code == 201
    recovery = approved.json()['recovery_id']
    assert clients['evaluator'].get(url).json()['recovery_id'] == recovery
    monkeypatch.setattr(HoldoutService, '_mark_read_started', original)
    response = clients['evaluator'].post(url + '/evaluate', json={'freeze_hash': frozen, 'recovery_id': recovery})
    assert response.status_code == 200, response.text
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM blind_oot_evaluations').fetchone()[0] == 1
        assert conn.execute('SELECT consumed_at FROM blind_oot_recoveries').fetchone()[0] is not None


def test_different_task_or_cohort_label_cannot_reblind_an_exposed_or_sealed_source(custody, tmp_path):
    app, clients, _, _ = custody
    raw, payload, _ = _seal(custody)
    source = app.state.settings.workspace / 'other-empty'
    source.mkdir()
    task = clients['custodian'].post('/api/tasks', json={'model_name':'Second synthetic task','validator':'qa','source_dir':str(source),'task_type':'modeling','run_mode':'manual'}).json()['id']
    payload['data_identity'] = 'another-request-and-cohort-label'
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort', files={'file':('same.parquet',raw,'application/octet-stream')}, data={'contract':json.dumps(payload)})
    assert response.status_code == 409 and response.json()['detail']['code'] == 'holdout_identity_already_reserved'
    public = tmp_path / 'exposed.parquet'
    different = _bytes(_frame(n=60, seed=819).drop(columns='split'))
    public.write_bytes(different)
    backend = DataBackend(app.state.settings.datasets_dir)
    registry = DatasetRegistry(DatasetRepository(app.state.settings.db_path), backend, app.state.settings.datasets_dir)
    # Exposure in an unrelated normal task still disqualifies the bytes.
    ordinary = clients['custodian'].post('/api/tasks', json={'model_name':'Ordinary synthetic exposure','validator':'qa','source_dir':str(source),'task_type':'modeling','run_mode':'manual'}).json()['id']
    registry.register_from_upload(ordinary, public)
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort', files={'file':('same.parquet',different,'application/octet-stream')}, data={'contract':json.dumps(payload)})
    assert response.status_code == 409 and response.json()['detail']['code'] == 'already_exposed_source_cannot_be_blind'


def test_mixed_development_and_time_oot_plans_are_rejected(custody):
    app, clients, _, task = custody
    _seal(custody)
    mixed = _frame()
    mixed.loc[len(mixed)-1, 'split'] = 'oot'
    response = clients['developer'].post(f'/api/tasks/{task}/blind-oot/development',files={'file':('mixed.parquet',_bytes(mixed),'application/octet-stream')})
    assert response.status_code == 422
    inputs, _ = _develop(custody)
    inputs['split_config'] = {'oot_by_time':'month','oot_size':0.2}
    blocked = app.state.tool_runner.invoke(ToolRef('modeling', 'prepare_modeling_frame'), inputs, task_id=task)
    assert not blocked.ok
    assert 'blind_development_cannot_create_oot_partition' in blocked.error
    from marvis.agent.modeling_setup import build_modeling_proposal
    backend = DataBackend(app.state.settings.datasets_dir)
    registry = DatasetRegistry(DatasetRepository(app.state.settings.db_path), backend, app.state.settings.datasets_dir)
    source = app.state.settings.workspace / 'empty-source'
    proposal = build_modeling_proposal(registry, backend, task, source, recipe='lr', target_col='y')
    assert 'oot' not in proposal.split_values and not proposal.holdout_values
    assert any('盲 OOT' in note for note in proposal.notes)


def test_missing_native_producer_receipt_fails_closed_before_freeze(custody):
    app, clients, _, task = custody
    _seal(custody)
    _, output = _develop(custody)
    with connect(app.state.settings.db_path) as conn:
        conn.execute("DELETE FROM task_artifacts WHERE task_id=? AND kind='modeling_native_model_closure'", (task,))
    response = clients['developer'].post(f'/api/tasks/{task}/blind-oot/freeze', json={'artifact_id': output['artifact_id']})
    assert response.status_code == 409, response.text
    assert response.json()['detail']['code'] == 'native_model_receipt_missing_or_invalid'
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM blind_oot_freezes').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM blind_oot_evaluations').fetchone()[0] == 0


def test_revoked_checker_cannot_reserve_or_read_and_resource_profile_decodes_no_rows(custody, monkeypatch):
    app, clients, ids, task = custody
    import marvis.modeling_holdout.service as service_module
    monkeypatch.setattr(service_module, 'MAX_SEALED_ROWS', 2)
    raw = _bytes(_frame(n=90).drop(columns='split'))
    payload = {'data_identity':'oversized-synthetic', 'developer_id':ids['developer'], 'evaluator_id':ids['evaluator'],
        'target_col':'y','split_col':'split','authority_basis':'synthetic mechanism test'}
    def denied(*args, **kwargs):
        raise AssertionError('Intake must not decode sample rows')
    original = service_module.pq.read_table
    monkeypatch.setattr(service_module.pq, 'read_table', denied)
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',files={'file':('raw.parquet',raw,'application/octet-stream')}, data={'contract':json.dumps(payload)})
    assert response.status_code == 413 and response.json()['detail']['code'] == 'holdout_resource_profile_exceeded'
    monkeypatch.setattr(service_module.pq, 'read_table', original)
    monkeypatch.setattr(service_module, 'MAX_SEALED_ROWS', 1_000_000)
    _seal(custody)
    _, output = _develop(custody)
    frozen = _freeze(custody, output)
    with connect(app.state.settings.db_path) as conn:
        conn.execute("UPDATE production_principals SET status='revoked' WHERE local_principal_id=?", (ids['evaluator'],))
    response = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate',json={'freeze_hash':frozen})
    assert response.status_code == 403
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM blind_oot_evaluations').fetchone()[0] == 0


def test_real_train_fitted_preprocessing_is_frozen_and_replayed_on_raw_holdout(custody):
    app, clients, _, task = custody
    _seal(custody)
    inputs, _ = _develop(custody)
    transformed = app.state.tool_runner.invoke(ToolRef('feature','normalize'),
        {'dataset_id':inputs['dataset_id'],'columns':['x','z'],'method':'zscore','split_col':'split'},task_id=task)
    assert transformed.ok, transformed.error
    assert transformed.output['fit_rows'] == 160
    inputs['dataset_id'] = transformed.output['result_dataset_id']
    trained = app.state.tool_runner.invoke(ToolRef('modeling','train_model'),inputs,task_id=task)
    assert trained.ok, trained.error
    frozen = _freeze(custody, trained.output)
    private = app.state.settings.workspace / '.blind_oot'
    closure = json.loads((private/'freezes'/frozen/'closure.json').read_text())
    assert closure['artifact']['params']['preprocessing_steps']
    from marvis.feature.preprocessing import sidecar_path
    registry = DatasetRegistry(DatasetRepository(app.state.settings.db_path),DataBackend(app.state.settings.datasets_dir),app.state.settings.datasets_dir)
    sidecar_path(registry.resolve_path(inputs['dataset_id'])).write_text('{"steps": []}')
    result = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate',json={'freeze_hash':frozen})
    assert result.status_code == 200, result.text
    assert np.isfinite(result.json()['metrics']['auc'])
    assert json.loads((private/'freezes'/frozen/'closure.json').read_text()) == closure


def test_unregistered_backend_read_is_durable_exposure_before_blind_intake(custody, tmp_path):
    app, clients, ids, task = custody
    raw = _bytes(_frame(n=90, seed=181).drop(columns='split'))
    public = tmp_path/'previously-read.parquet'
    public.write_bytes(raw)
    assert len(DataBackend(app.state.settings.datasets_dir).read_frame(public)) == 90
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT 1 FROM blind_oot_exposures WHERE content_hash=?',(hashlib.sha256(raw).hexdigest(),)).fetchone()
    payload = {'data_identity':'previous-read','developer_id':ids['developer'],'evaluator_id':ids['evaluator'],'target_col':'y','split_col':'split','authority_basis':'synthetic'}
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',files={'file':('raw.parquet',raw,'application/octet-stream')},data={'contract':json.dumps(payload)})
    assert response.status_code == 409 and response.json()['detail']['code'] == 'already_exposed_source_cannot_be_blind'


def test_semantic_reencoding_reordering_and_subsets_cannot_create_new_budget_or_enter_training(custody, tmp_path):
    app, clients, _, task = custody
    _, payload, _ = _seal(custody)
    frame = _frame(n=90,seed=181).drop(columns='split')
    for index, alternate in enumerate((frame.sample(frac=1,random_state=8), frame.iloc[:30], frame.assign(irrelevant_padding=17))):
        stream = BytesIO()
        alternate.to_parquet(stream,index=False,compression='gzip')
        raw = stream.getvalue()
        source=app.state.settings.workspace/f'new-empty-{index}'
        source.mkdir()
        other = clients['custodian'].post('/api/tasks',json={'model_name':'Clone attempt','validator':'qa','source_dir':str(source),'task_type':'modeling','run_mode':'manual'}).json()['id']
        payload['data_identity']=f'changed-label-{index}'
        response = clients['custodian'].post(f'/api/tasks/{other}/blind-oot/cohort',files={'file':('reencoded.parquet',raw,'application/octet-stream')},data={'contract':json.dumps(payload)})
        assert response.status_code == 409 and response.json()['detail']['code'] == 'holdout_identity_already_reserved'
        public=tmp_path/f'reencoded-{index}.parquet'
        public.write_bytes(raw)
        with pytest.raises(HoldoutError,match='sealed_holdout_access_forbidden'):
            DataBackend(app.state.settings.datasets_dir).read_frame(public)
    inputs, output = _develop(custody)
    assert clients['other'].get(f"/api/datasets/{inputs['dataset_id']}/preview").status_code == 403
    assert clients['developer'].get(f"/api/datasets/{inputs['dataset_id']}/preview").status_code == 200
    assert clients['evaluator'].get(f"/api/artifacts/{task}/modeling_artifacts/anything.joblib").status_code == 403
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM blind_oot_cohorts').fetchone()[0] == 1
    _freeze(custody, output)


def test_register_private_path_denies_before_any_public_upload_copy(custody, monkeypatch):
    app, clients, _, task = custody
    _seal(custody)
    original=next((app.state.settings.workspace/'.blind_oot').glob('sealed/*/original.parquet'))
    import marvis.routers.data as data_router
    copies=[]
    def copying(*args, **kwargs):
        copies.append(args)
        raise AssertionError('Private custody bytes must not be staged')
    monkeypatch.setattr(data_router,'_copy_local_dataset_path',copying)
    response=clients['developer'].post(f'/api/tasks/{task}/datasets/register-path',json={'path':str(original),'role':'sample'})
    assert response.status_code == 403, response.text
    assert not copies



def test_other_cohort_split_name_does_not_disable_existing_identity_projection(custody, tmp_path):
    app, clients, _, _ = custody
    _, payload, _ = _seal(custody)
    source=app.state.settings.workspace/'different-split-empty'
    source.mkdir()
    other=clients['custodian'].post('/api/tasks',json={'model_name':'Different split','validator':'qa','source_dir':str(source),'task_type':'modeling','run_mode':'manual'}).json()['id']
    payload['data_identity']='different-split-authority'
    payload['split_col']='x'
    response=clients['custodian'].post(f'/api/tasks/{other}/blind-oot/cohort',files={'file':('fresh.parquet',_bytes(_frame(n=90,seed=673).drop(columns='split')),'application/octet-stream')},data={'contract':json.dumps(payload)})
    assert response.status_code == 201, response.text
    raw=BytesIO()
    _frame(n=90,seed=181).drop(columns='split').to_parquet(raw,index=False,compression='gzip')
    path=tmp_path/'reencoded-original.parquet'
    path.write_bytes(raw.getvalue())
    with pytest.raises(HoldoutError,match='sealed_holdout_access_forbidden'):
        DataBackend(app.state.settings.datasets_dir).read_frame(path)


@pytest.mark.parametrize('suffix', ['.csv', '.feather', '.parquet', '.xlsx'])
def test_normal_format_and_unsupported_padding_cannot_return_sealed_rows(custody, tmp_path, suffix):
    app, _, _, _ = custody
    _seal(custody)
    original = _frame(n=90, seed=181).drop(columns='split')
    padded = original.assign(irrelevant_nested_padding=[{'padding': 1}] * len(original))
    path = tmp_path / f'ordinary-alias{suffix}'
    if suffix == '.csv':
        original.to_csv(path, index=False)
    elif suffix == '.feather':
        padded.to_feather(path)
    elif suffix == '.xlsx':
        original.to_excel(path, index=False)
    else:
        padded.to_parquet(path, index=False, compression='gzip')
    with pytest.raises(HoldoutError, match='sealed_holdout_access_forbidden'):
        DataBackend(app.state.settings.datasets_dir).read_frame(path) if suffix != '.xlsx' else DatasetRegistry(
            DatasetRepository(app.state.settings.db_path), DataBackend(app.state.settings.datasets_dir),
            app.state.settings.datasets_dir).register_from_upload('ordinary', path, role='sample')


@pytest.mark.parametrize('suffix', ['.csv', '.feather', '.parquet', '.xlsx'])
def test_public_format_and_unsupported_padding_prevents_later_reblinding(custody, tmp_path, suffix):
    app, clients, ids, task = custody
    original = _frame(n=90, seed=181).drop(columns='split')
    padded = original.assign(irrelevant_nested_padding=[{'padding': 1}] * len(original))
    path = tmp_path / f'public-first{suffix}'
    if suffix == '.csv':
        original.to_csv(path, index=False)
    elif suffix == '.feather':
        padded.to_feather(path)
    elif suffix == '.xlsx':
        original.to_excel(path, index=False)
    else:
        padded.to_parquet(path, index=False, compression='gzip')
    from marvis.modeling_holdout.guards import record_exposure
    if suffix == '.xlsx':
        record_exposure(app.state.settings.db_path, app.state.settings.workspace, path)
    else:
        assert len(DataBackend(app.state.settings.datasets_dir).read_frame(path)) == 90
    payload = {'data_identity': 'ordinary-first', 'developer_id': ids['developer'], 'evaluator_id': ids['evaluator'],
               'target_col': 'y', 'split_col': 'split', 'authority_basis': 'synthetic'}
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',
        files={'file': ('fresh.parquet', _bytes(original), 'application/octet-stream')},
        data={'contract': json.dumps(payload)})
    assert response.status_code == 409, response.text
    assert response.json()['detail']['code'] == 'already_exposed_source_cannot_be_blind'


def test_unverifiable_known_projection_fails_closed_and_unknown_past_exposure_blocks_blind_claim(custody, tmp_path, monkeypatch):
    app, clients, ids, task = custody
    from marvis.modeling_holdout import identity
    original = _frame(n=90, seed=181).drop(columns='split')
    path = tmp_path / 'resource-limited.parquet'
    path.write_bytes(_bytes(original))
    monkeypatch.setattr(identity, 'MAX_SEALED_ROWS', 30)
    assert len(DataBackend(app.state.settings.datasets_dir).read_frame(path)) == 90
    monkeypatch.setattr(identity, 'MAX_SEALED_ROWS', 1_000_000)
    payload = {'data_identity': 'unknown-exposure', 'developer_id': ids['developer'], 'evaluator_id': ids['evaluator'],
               'target_col': 'y', 'split_col': 'split', 'authority_basis': 'synthetic'}
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',
        files={'file': ('reencoded.parquet', _bytes(original.sample(frac=1, random_state=4)), 'application/octet-stream')},
        data={'contract': json.dumps(payload)})
    assert response.status_code == 409 and response.json()['detail']['code'] == 'prior_exposure_provenance_unknown'

    # A fresh independent schema can still be sealed. The existing unverified
    # x/z/y source does not restrict unrelated development workflows.
    original = original.rename(columns={'x': 'a', 'z': 'b'})
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',
        files={'file': ('independent.parquet', _bytes(original), 'application/octet-stream')},
        data={'contract': json.dumps(payload)})
    assert response.status_code == 201, response.text
    monkeypatch.setattr(identity, 'MAX_SEALED_ROWS', 30)
    path = tmp_path / 'known-profile-unverifiable.parquet'
    path.write_bytes(_bytes(original.sample(frac=1, random_state=7)))
    with pytest.raises(HoldoutError, match='holdout_admission_identity_unverifiable'):
        DataBackend(app.state.settings.datasets_dir).read_frame(path)


def test_native_validation_scan_refuses_sealed_csv_alias_before_preview(custody, tmp_path, monkeypatch):
    app, clients, _, _ = custody
    _seal(custody)
    source = app.state.settings.workspace / 'ordinary-validation'
    source.mkdir()
    _frame(n=90, seed=181).drop(columns='split').to_csv(source / 'sample.csv', index=False)
    import nbformat
    nbformat.write(nbformat.v4.new_notebook(cells=[]), source / 'model.ipynb')
    (source / 'model.pmml').write_text('<PMML/>')
    (source / 'dictionary.csv').write_text('feature,importance\nx,1\n')
    response = clients['custodian'].post('/api/tasks', json={'model_name': 'Ordinary native validation',
        'validator': 'qa', 'source_dir': str(source), 'run_mode': 'manual', 'task_type': 'validation',
        'notebook_path': 'model.ipynb', 'sample_path': 'sample.csv', 'pmml_path': 'model.pmml',
        'dictionary_path': 'dictionary.csv'})
    assert response.status_code == 200, response.text
    calls = []
    def inspecting(*args, **kwargs):
        calls.append(args)
        raise AssertionError('Native validation must not preview reserved rows')
    monkeypatch.setattr('marvis.api_scan_helpers.inspect_sample_schema', inspecting)
    response = clients['custodian'].post(f"/api/tasks/{response.json()['id']}/scan")
    assert response.status_code == 403, response.text
    assert response.json()['detail']['code'] == 'sealed_holdout_access_forbidden'
    assert not calls


@pytest.mark.parametrize('crash_point', ['before_read', 'after_read'])
def test_real_evaluator_process_death_retains_budget_across_restart(custody, tmp_path, crash_point):
    import subprocess
    import sys
    app, clients, ids, task = custody
    _seal(custody)
    _, output = _develop(custody)
    frozen = _freeze(custody, output)
    script = tmp_path / 'evaluator-process.py'
    script.write_text('''import json, os, sys
from pathlib import Path
from marvis.settings import Settings
from marvis.modeling_holdout.service import HoldoutService
from marvis.modeling_holdout.contracts import Evaluate
settings = Settings.from_workspace(sys.argv[1])
if sys.argv[5] == 'before_read':
    HoldoutService._mark_read_started = lambda *args: os._exit(87)
else:
    original = HoldoutService._score_sealed
    def native_score_then_die(self, cohort, closure, scorer):
        result = original(self, cohort, closure, scorer)
        proof = self._path('process-death-proof.json')
        proof.write_text(json.dumps({'rows': result['rows']}))
        proof.chmod(0o600)
        os._exit(87)
    HoldoutService._score_sealed = native_score_then_die
HoldoutService(settings).evaluate(sys.argv[2], Evaluate(freeze_hash=sys.argv[3]), sys.argv[4])
''')
    process = subprocess.run([sys.executable, '-c', script.read_text(), str(app.state.settings.workspace), task, frozen,
                              ids['evaluator'], crash_point], capture_output=True, timeout=30)
    assert process.returncode == 87, process.stderr.decode()
    url = f'/api/tasks/{task}/blind-oot'
    summary = clients['custodian'].get(url).json()
    assert summary['state'] == ('prepared' if crash_point == 'before_read' else 'reading')
    assert summary['sample_read_started'] == (crash_point == 'after_read')
    response = clients['evaluator'].post(url + '/evaluate', json={'freeze_hash': frozen, 'request_id': 'new-after-restart'})
    assert response.status_code == 409
    recovery = clients['custodian'].post(url + '/recoveries', json={'mode': 'resume_before_read', 'reason': 'Authorized synthetic process-death acceptance'})
    if crash_point == 'before_read':
        assert recovery.status_code == 201, recovery.text
        assert clients['evaluator'].post(url + '/evaluate', json={'freeze_hash': frozen, 'recovery_id': recovery.json()['recovery_id']}).status_code == 200
    else:
        assert json.loads((app.state.settings.workspace / '.blind_oot/process-death-proof.json').read_text()) == {'rows': 90}
        assert recovery.status_code == 409 and recovery.json()['detail']['code'] == 'holdout_cannot_repeat_sample_read'
        assert clients['custodian'].post(url + '/recoveries', json={'mode': 'publish_result', 'reason': 'No result available'}).status_code == 409
    with connect(app.state.settings.db_path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM blind_oot_evaluations').fetchone()[0] == 1
