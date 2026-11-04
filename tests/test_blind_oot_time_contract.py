"""Blind OOT time contract: fact collection, conclusion gating and sealed-read isolation."""
# ruff: noqa: F811 -- tests take the shared custody fixture as a parameter

import json
from pathlib import Path
import subprocess
import sys

import pytest

from marvis import time_facts
from marvis.db_schema import connect, init_db
from marvis.local_execution import LOOPBACK_ONLY, execution_capability, execution_command
from marvis.modeling_holdout import sealed_worker
from marvis.modeling_holdout.service import with_conclusions
from marvis.packs.labeling.tools import tool_define_label
from tests.test_labeling_workflow import _labeling_runtime
from tests.test_modeling_holdout import _bytes, _develop, _frame, _freeze, custody  # noqa: F401 -- fixture

ROOT = Path(__file__).resolve().parents[1]
darwin_sandbox = pytest.mark.skipif(sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').exists(),
                                    reason='macOS Seatbelt backend only')


def _seal_with(custody, **facts):
    app, clients, ids, task = custody
    payload = {'data_identity': 'synthetic-time-contract-cohort', 'developer_id': ids['developer'],
               'evaluator_id': ids['evaluator'], 'target_col': 'y', 'split_col': 'split',
               'target_type': 'binary', 'authority_basis': 'Authorized synthetic fixture', **facts}
    raw = _bytes(_frame('binary', n=90, seed=181).drop(columns='split'))
    response = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',
        files={'file': ('independent.parquet', raw, 'application/octet-stream')},
        data={'contract': json.dumps(payload)})
    return raw, payload, response


def _develop_with(custody, **facts):
    app, clients, _, task = custody
    response = clients['developer'].post(f'/api/tasks/{task}/blind-oot/development',
        files={'file': ('development.parquet', _bytes(_frame('binary')), 'application/octet-stream')}, data=facts)
    assert response.status_code == 201, response.text
    return response.json()


def _evaluate(custody, output):
    _, clients, _, task = custody
    frozen = _freeze(custody, output)
    response = clients['evaluator'].post(f'/api/tasks/{task}/blind-oot/evaluate', json={'freeze_hash': frozen})
    assert response.status_code == 200, response.text
    return response.json()


def test_unknown_label_time_marks_blind_result_not_an_acceptance_conclusion(custody):
    _, clients, _, task = custody
    assert _seal_with(custody)[2].status_code == 201
    _, output = _develop(custody)
    result = _evaluate(custody, output)
    gate = result['conclusions']['blind_oot_acceptance']
    assert gate['status'] == 'blocked'
    assert gate['label'] == '不能作为验收结论'
    assert {'label_available_at_unknown', 'candidate_provenance_unknown'} <= set(gate['reasons'])
    assert result['label_maturity'] == 'unknown'
    assert result['time_facts']['label_available_at'] == {'state': 'unknown', 'value': 'unknown'}
    assert set(result['metrics']) == {'rows', 'ks', 'auc'}
    assert clients['evaluator'].get(f'/api/tasks/{task}/blind-oot/result').json() == result


def test_recorded_facts_allow_acceptance_and_are_immutable(custody):
    app, clients, _, task = custody
    raw, payload, response = _seal_with(custody, label_available_at='2020-01-31')
    assert response.status_code == 201, response.text
    assert response.json()['time_contract']['facts']['label_available_at']['value'] == '2020-01-31'
    # Re-sealing the same cohort with another label time cannot rewrite the fact.
    conflict = clients['custodian'].post(f'/api/tasks/{task}/blind-oot/cohort',
        files={'file': ('independent.parquet', raw, 'application/octet-stream')},
        data={'contract': json.dumps({**payload, 'label_available_at': '2021-01-31'})})
    assert conflict.status_code == 409 and conflict.json()['detail']['code'] == 'time_fact_immutable'
    dev = _develop_with(custody, historical_parameters='existed', candidate_provenance='pre_registered')
    assert dev['time_contract']['conclusions']['blind_oot_acceptance']['status'] == 'allowed'
    inputs = {'dataset_id': dev['dataset_id'], 'recipe': 'lr', 'features': ['x', 'z'], 'target_col': 'y',
              'split_col': 'split', 'split_values': {'train': 'train', 'test': 'test'}, 'seed': 19,
              'params': {}, 'target_type': 'binary'}
    from marvis.plugins.manifest import ToolRef
    trained = app.state.tool_runner.invoke(ToolRef('modeling', 'train_model'), inputs, task_id=task)
    assert trained.ok, trained.error
    result = _evaluate(custody, trained.output)
    assert result['label_maturity'] == 'mature'
    assert result['conclusions']['blind_oot_acceptance'] == {
        'status': 'allowed', 'depends_on': ['label_available_at', 'candidate_provenance'], 'reasons': [], 'label': None}
    assert result['conclusions']['historical_period_comparison']['status'] == 'allowed'
    summary = clients['developer'].get(f'/api/tasks/{task}/blind-oot').json()
    assert summary['label_maturity'] == 'mature'
    assert 'metrics' not in json.dumps(summary)
    with connect(app.state.settings.db_path) as conn, pytest.raises(Exception):
        conn.execute("UPDATE time_facts SET value='2030-01-01'")


def test_holdout_informed_candidates_and_invalid_values_are_handled(custody):
    _, clients, _, task = custody
    bad = _seal_with(custody, label_available_at='not-a-date')[2]
    assert bad.status_code == 422 and bad.json()['detail']['code'] == 'time_fact_value_invalid'
    assert _seal_with(custody, label_available_at='2020-01-31')[2].status_code == 201
    invalid = clients['developer'].post(f'/api/tasks/{task}/blind-oot/development',
        files={'file': ('development.parquet', _bytes(_frame('binary')), 'application/octet-stream')},
        data={'candidate_provenance': 'guessed'})
    assert invalid.status_code == 422
    dev = _develop_with(custody, candidate_provenance='holdout_informed')
    gate = dev['time_contract']['conclusions']['blind_oot_acceptance']
    assert gate['status'] == 'blocked' and gate['reasons'] == ['candidate_informed_by_holdout']
    # Ordinary data-intake endpoint cannot bypass the blind role split.
    denied = clients['developer'].post(f'/api/tasks/{task}/time-facts', json={
        'fact': 'candidate_provenance', 'value': 'pre_registered', 'subject_kind': 'task'})
    assert denied.status_code == 409


@darwin_sandbox
def test_sealed_rows_are_scored_only_in_isolated_worker_with_unchanged_metrics(custody, monkeypatch):
    calls = []
    original_init, original_score = sealed_worker.IsolatedSession.__init__, sealed_worker.IsolatedSession.score

    def init(self, path, sealed_hash, runtime_dir, artifact, contract):
        self._test_args = (path, runtime_dir, artifact, contract)
        original_init(self, path, sealed_hash, runtime_dir, artifact, contract)

    def observe(self):
        isolated = original_score(self)
        import pandas as pd
        path, runtime_dir, artifact, contract = self._test_args
        scorer = sealed_worker._scorer(artifact, runtime_dir)
        in_process = sealed_worker.sealed_metrics(pd.read_parquet(path), contract,
                                                  artifact['params'].get('classes'), scorer)
        calls.append((isolated, in_process))
        return isolated

    monkeypatch.setattr(sealed_worker.IsolatedSession, '__init__', init)
    monkeypatch.setattr(sealed_worker.IsolatedSession, 'score', observe)
    assert _seal_with(custody)[2].status_code == 201
    _, output = _develop(custody)
    result = _evaluate(custody, output)
    assert result['sealed_read_boundary'] == 'isolated_worker'
    assert len(calls) == 1 and calls[0][0] == calls[0][1] == result['metrics']


@darwin_sandbox
def test_isolated_worker_profile_cannot_read_workspace_files(tmp_path):
    from marvis.plugins import native_isolation as isolation
    secret = tmp_path / 'workspace' / 'other.txt'
    secret.parent.mkdir()
    secret.write_text('secret')
    allowed = tmp_path / 'allowed.txt'
    allowed.write_text('ok')
    runtime = isolation._runtime(sys.executable)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    profile = isolation._profile(runtime, scratch.resolve(), [str(ROOT / 'marvis')], [str(allowed.resolve())])
    script = f"import pathlib; print(pathlib.Path({str(allowed.resolve())!r}).read_text()); pathlib.Path({str(secret.resolve())!r}).read_text()"
    completed = subprocess.run(['/usr/bin/sandbox-exec', '-p', profile, runtime['python'], '-I', '-B', '-c', script],
                               capture_output=True, text=True, timeout=30)
    assert completed.stdout.strip() == 'ok'
    assert completed.returncode != 0 and 'PermissionError' in completed.stderr


@darwin_sandbox
def test_loopback_notebook_tree_is_denied_the_sealed_directory(tmp_path):
    if not execution_capability(LOOPBACK_ONLY).available:
        pytest.skip('loopback backend unavailable')
    sealed = tmp_path / '.blind_oot' / 'sealed' / 'original.parquet'
    sealed.parent.mkdir(parents=True)
    sealed.write_bytes(b'sealed')
    public = tmp_path / 'public.csv'
    public.write_text('public')
    command = execution_command(['/bin/sh', '-c', f'cat {public}; cat {sealed}'], LOOPBACK_ONLY)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.stdout == 'public'
    assert completed.returncode != 0 and 'not permitted' in completed.stderr
    # Hard link, clone, rename and symlink-follow cannot carry it out of the folder either.
    for name, step in (('hard', f'ln {sealed} {tmp_path}/hard'), ('clone', f'cp -c {sealed} {tmp_path}/clone'),
                       ('moved', f'mv {sealed.parent.parent} {tmp_path}/moved'),
                       ('sym', f'ln -s {sealed} {tmp_path}/sym && cat {tmp_path}/sym')):
        attempt = subprocess.run(execution_command(['/bin/sh', '-c', step], LOOPBACK_ONLY),
                                 capture_output=True, text=True, timeout=30)
        assert attempt.returncode != 0, name
        assert not (tmp_path / name).exists() or (tmp_path / name).is_symlink(), name
    assert sealed.read_bytes() == b'sealed'


@darwin_sandbox
def test_worker_failure_before_read_keeps_budget_and_recovers(custody, monkeypatch):
    from marvis.plugins import native_isolation as isolation
    _, clients, _, task = custody
    original = isolation._profile
    # The worker cannot read the pinned model files, so it fails while loading.
    monkeypatch.setattr(isolation, '_profile', lambda runtime, root, packages, files: original(runtime, root, packages, ()))
    assert _seal_with(custody)[2].status_code == 201
    _, output = _develop(custody)
    frozen = _freeze(custody, output)
    url = f'/api/tasks/{task}/blind-oot'
    failed = clients['evaluator'].post(url + '/evaluate', json={'freeze_hash': frozen})
    assert failed.status_code == 409 and failed.json()['detail']['code'] == 'holdout_isolated_worker_not_ready'
    summary = clients['custodian'].get(url).json()
    assert summary['sample_read_started'] is False
    assert 'resume_before_read' in summary['recovery_options']
    monkeypatch.setattr(isolation, '_profile', original)
    recovery = clients['custodian'].post(url + '/recoveries', json={'mode': 'resume_before_read', 'reason': 'Worker fixed'})
    assert recovery.status_code == 201, recovery.text
    done = clients['evaluator'].post(url + '/evaluate', json={'freeze_hash': frozen, 'recovery_id': recovery.json()['recovery_id']})
    assert done.status_code == 200, done.text
    assert done.json()['sealed_read_boundary'] == 'isolated_worker'


def test_pre_intake_and_untrusted_task_facts_cannot_satisfy_blind_acceptance(custody):
    app, clients, ids, task = custody
    # Facts posted before custody intake make the task ineligible for blind mode.
    early = clients['other'].post(f'/api/tasks/{task}/time-facts', json={
        'fact': 'candidate_provenance', 'value': 'pre_registered', 'subject_kind': 'task'})
    assert early.status_code == 201, early.text
    refused = _seal_with(custody, label_available_at='2020-01-31')[2]
    assert refused.status_code == 409 and refused.json()['detail']['code'] == 'blind_intake_must_precede_development'


def test_facts_from_other_sources_or_actors_count_as_absent(custody):
    app, clients, ids, task = custody
    response = _seal_with(custody, label_available_at='2020-01-31')[2]
    assert response.status_code == 201
    cohort = response.json()['cohort_id']
    with connect(app.state.settings.db_path) as conn:
        time_facts.record_fact_on_connection(conn, task_id=task, subject_kind='task', subject_id=task,
            fact='candidate_provenance', value='pre_registered', source='holdout_development', actor=ids['other'])
        time_facts.record_fact_on_connection(conn, task_id=task, subject_kind='task', subject_id=task,
            fact='historical_parameters', value='existed', source='data_intake', actor=ids['developer'])
    summary = clients['developer'].get(f'/api/tasks/{task}/blind-oot').json()
    facts = summary['time_contract']['facts']
    assert facts['candidate_provenance']['state'] == facts['historical_parameters']['state'] == 'unknown'
    assert facts['label_available_at']['recorded_by'] == ids['custodian']
    assert summary['time_contract']['conclusions']['blind_oot_acceptance']['reasons'] == ['candidate_provenance_unknown']
    assert cohort
    assert clients['developer'].get(f'/api/tasks/{task}/time-facts').status_code == 409


def test_label_fact_conflict_becomes_labeling_contract_error(tmp_path, monkeypatch):
    from marvis.packs.labeling.contracts import LabelingContractError
    settings, registry, backend, workspace_repo, task, ctx, request = _labeling_runtime(tmp_path)

    def conflict(conn, **values):
        raise time_facts.TimeFactError('time_fact_immutable', 409)

    monkeypatch.setattr(time_facts, 'record_fact_on_connection', conflict)
    with pytest.raises(LabelingContractError, match='time_fact_immutable'):
        tool_define_label({**request.to_dict(), 'proposal_hash': request.contract_hash,
                           'confirm_immature_cohorts': False}, ctx)


def test_api_process_fallback_is_labelled(custody, monkeypatch):
    monkeypatch.setattr(sealed_worker, 'available_boundary', lambda: sealed_worker.API_PROCESS)
    assert _seal_with(custody)[2].status_code == 201
    _, output = _develop(custody)
    assert _evaluate(custody, output)['sealed_read_boundary'] == 'api_process'


def test_label_construction_records_label_available_time(tmp_path):
    settings, registry, backend, workspace_repo, task, ctx, request = _labeling_runtime(tmp_path)
    output = tool_define_label({**request.to_dict(), 'proposal_hash': request.contract_hash,
                                'confirm_immature_cohorts': False}, ctx)
    with connect(settings.db_path) as conn:
        facts = time_facts.facts_for(conn, (('dataset', output['result_dataset_id']),))
    assert facts['label_available_at']['value'] == '2026-07-31'
    assert facts['label_available_at']['assurance'] == 'platform_derived'
    assert facts['historical_parameters']['state'] == 'unknown'


def test_data_intake_endpoint_records_and_lists_task_facts(tmp_path):
    from fastapi.testclient import TestClient
    from marvis.app import create_app
    app = create_app(tmp_path / 'workspace')
    with TestClient(app) as client:
        source = app.state.settings.workspace / 'src'
        source.mkdir()
        task = client.post('/api/tasks', json={'model_name': 'facts', 'validator': 'qa', 'source_dir': str(source),
                                               'task_type': 'modeling', 'run_mode': 'manual', 'recipes': ['lr']}).json()['id']
        url = f'/api/tasks/{task}/time-facts'
        assert client.get(url).json()['facts'] == []
        made = client.post(url, json={'fact': 'historical_parameters', 'value': 'absent', 'subject_kind': 'task'})
        assert made.status_code == 201, made.text
        assert client.post(url, json={'fact': 'historical_parameters', 'value': 'existed',
                                      'subject_kind': 'task'}).status_code == 409
        assert client.post(url, json={'fact': 'label_available_at', 'value': '2020-01-01',
                                      'subject_kind': 'dataset', 'subject_id': 'missing'}).status_code == 404
        listed = client.get(url).json()
        assert [(f['fact'], f['value'], f['source']) for f in listed['facts']] == [('historical_parameters', 'absent', 'data_intake')]
        assert listed['conclusion_dependencies']['blind_oot_acceptance'] == ['label_available_at', 'candidate_provenance']


def test_gate_treats_late_and_immature_facts_as_not_established(tmp_path):
    db = tmp_path / 'db.sqlite'
    init_db(db)
    with connect(db) as conn:
        time_facts.record_fact_on_connection(conn, task_id='t', subject_kind='blind_cohort', subject_id='c',
            fact='label_available_at', value='2030-06-30', source='holdout_custody', actor='a')
        time_facts.record_fact_on_connection(conn, task_id='t', subject_kind='task', subject_id='t',
            fact='candidate_provenance', value='pre_registered', source='holdout_development', actor='a')
        subjects = (('blind_cohort', 'c'), ('task', 't'))
        late = time_facts.facts_for(conn, subjects, cutoff=0)
        current = time_facts.facts_for(conn, subjects)
    assert late['label_available_at']['state'] == 'recorded_after_read'
    late_gate = time_facts.gate_conclusions(late, 2e9)['conclusions']['blind_oot_acceptance']
    assert late_gate['reasons'] == ['label_available_at_recorded_after_read', 'candidate_provenance_recorded_after_read']
    early = time_facts.gate_conclusions(current, 1.8e9)  # 2027: labels not yet observable
    assert early['label_maturity'] == 'immature'
    assert early['conclusions']['blind_oot_acceptance']['reasons'] == ['labels_immature']
    assert time_facts.gate_conclusions(current, 2e9)['conclusions']['blind_oot_acceptance']['status'] == 'allowed'
    # Signed results from before this contract carry no facts and stay blocked.
    assert with_conclusions({'metrics': {'auc': 0.7}})['conclusions']['blind_oot_acceptance']['status'] == 'blocked'


def test_frontend_shows_blocked_conclusion_and_collects_fact_inputs():
    subprocess.run(['node', '--input-type=module', '-e', '''
        import assert from 'node:assert/strict';
        import { blindOotHtml } from './marvis/static/js/blind-oot-controller.js';
        const summary = {state:'committed', viewer_role:'evaluate', freeze_hash:'0'.repeat(64)};
        const result = {metrics:{auc:0.7}, time_facts:{label_available_at:{state:'unknown',value:'unknown'}},
          conclusions:{blind_oot_acceptance:{status:'blocked',label:'不能作为验收结论'}, historical_period_comparison:{status:'allowed',label:null}}};
        const html = blindOotHtml({summary, principal:{role:'checker'}, result});
        assert(html.includes('盲 OOT 结果：不能作为验收结论'));
        assert(!html.includes('历史同期对比：'));
        assert(html.includes('标签可得时间') && html.includes('未知'));
        const seal = blindOotHtml({summary:null, principal:{role:'admin'}, participants:[]});
        assert(seal.includes('name="label_available_at"'));
        const dev = blindOotHtml({summary:{state:'sealed', viewer_role:'develop'}, principal:{role:'maker'}});
        assert(dev.includes('name="candidate_provenance"') && dev.includes('name="historical_parameters"'));
    '''], cwd=ROOT, capture_output=True, text=True, check=True)
