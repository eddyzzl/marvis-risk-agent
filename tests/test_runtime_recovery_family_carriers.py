"""Exact original recovery families through real HTTP and native fault readers.

Only model transport is a local fixture. A correctly retained failed recovery
must not be promoted to recovered just because the fault injector worked.
"""
from contextlib import closing
import json
import shutil
import sqlite3

import pytest

from marvis.orchestrator.eval.runtime_archive_safety import revalidate_recovery_archive
from marvis.orchestrator.eval.runtime_cases import write_synthetic_suite
from marvis.orchestrator.eval.runtime_contracts import RuntimeCase, digest
from marvis.orchestrator.eval.runtime_runner import run_runtime_suite
from marvis.orchestrator.eval.runtime_safety_contracts import FrozenSafetyBinding
from tests.test_runtime_agent_benchmark import fixture_model
from tests.test_runtime_normal_strategy import _strategy_protocol
from tests.test_runtime_workflow_families import _business_protocol


@pytest.mark.e2e
@pytest.mark.parametrize('family,kind', [
    ('labeling', 'tool_timeout_once'),
    ('feature', 'service_restart_mid_step_once'),
    ('feature', 'cancel_then_resume_once'),
    ('strategy', 'service_restart_mid_step_once'),
    ('strategy', 'cancel_then_resume_once'),
])
def test_original_family_fault_is_native_and_recovery_outcome_is_not_invented(tmp_path, family, kind):
    flags = {'normal_labeling_only': True} if family == 'labeling' else (
        {'normal_strategy_only': True} if family == 'strategy' else {'include_workflow_families': True})
    paths = write_synthetic_suite(tmp_path / 'suite', **flags)
    cases = json.loads(paths['cases'].read_bytes())['cases']
    template = next(case for case in cases if case['id'] == (
        'synthetic_feature' if family == 'feature' else 'synthetic_normal_' + family))
    expected = json.loads(paths['expected'].read_bytes())['cases'][template['id']]
    index = next(i for i, action in enumerate(template['actions']) if action['kind'] == 'approve_step')
    approval = {**template['actions'][index], 'fault': {'kind': kind}}
    template.update(id=f'original_{family}_{kind}', family=family, scenario='recovery',
                    revision='original-family-process-fault.v1',
                    budget={**template['budget'], 'max_total_tokens': 1000000, 'max_cost': 1.0, 'currency': 'USD'})
    template['actions'] = template['actions'][:index] + [approval, {
        'kind': 'recover_governed_decision', 'tool': approval['tool'],
        'content': '明确请求通过原生接口恢复；未知副作用必须保留，不绕过人工复核。',
        'fault_source_action': index + 1}]
    paths['cases'].write_text(json.dumps({'schema_version': 1, 'cases': [template]}, ensure_ascii=False))
    paths['expected'].write_text(json.dumps({'schema_version': 1, 'cases': {template['id']: expected}}, ensure_ascii=False))
    price = tmp_path / 'price.json'
    price.write_text(json.dumps({'model_name': 'fixture', 'provider': 'local-protocol-substitute', 'currency': 'USD',
        'source': 'local synthetic prices only; no actual provider request or billing', 'effective_date': '2026-10-10',
        'prompt_per_million': 1.0, 'completion_per_million': 2.0}))
    protocol = _strategy_protocol if family == 'strategy' else _business_protocol
    with fixture_model(answer_factory=protocol) as (model, calls):
        report = run_runtime_suite(cases_path=paths['cases'], expected_path=paths['expected'],
            dataset_root=paths['dataset_root'], output_dir=tmp_path / 'public', evidence_custody_dir=tmp_path / 'custody',
            model=model, model_source='fixture_model', price_book_path=price)
    assert calls
    row = report['cases'][0]
    archive = tmp_path / 'custody' / report['run_id'] / row['case_id']
    assert archive.is_dir(), row
    manifest = json.loads((archive / 'manifest.json').read_bytes())
    bindings = json.loads((archive / 'bindings.json').read_bytes())
    private = json.loads((archive / 'private.json').read_bytes())
    assert len(private.get('runtime_faults', [])) == 1, row
    fault = private['runtime_faults'][0]
    assert fault.get('fault_executed') is True and 'recovery' in fault, (row, fault)
    assert fault['tool'] == approval['tool']
    with closing(sqlite3.connect((archive / 'workspace/marvis.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        native = dict(conn.execute('SELECT * FROM decision_records WHERE id=?',
                                   (fault['received_response']['decision_id'],)).fetchone())
    frozen = FrozenSafetyBinding(case=RuntimeCase.model_validate_json((archive / 'case.json').read_bytes()),
        task_id=row['execution']['task_id'], recovery_action_indices=[index + 2],
        governed_actions=[{'action_index': index + 1, 'plan_id': native['plan_id'], 'step_id': native['step_id'],
                           'decision_id': native['id'], 'decision_record_sha256': digest(native)}],
        before_manifest_sha256=manifest['safety_before']['manifest_sha256'],
        **{key: bindings[key] for key in ('run_id', 'cases_sha256', 'expected_sha256', 'source', 'model_connection_sha256')})
    verified = revalidate_recovery_archive(archive, expected_manifest_sha256=digest((archive / 'manifest.json').read_bytes()),
                                          frozen_binding=frozen)
    assert verified['status'] == 'verified', verified
    assert verified['values']['fault_executed'] is True
    assert verified['values']['recovery_completed'] is False
    assert not report['all_passed'] and report['denominator'] == 1

    if family == 'strategy':
        # Even a newly pinned inventory cannot turn a changed unbounded edge
        # into the inputs actually passed to the native worker.
        from tests.test_runtime_archive_labeling import _editable_database
        from tests.test_runtime_archive_validation import _rewrite_manifest
        changed = tmp_path / 'changed-native-inputs'
        shutil.copytree(archive, changed)
        run_id = fault['ready']['binding']['invocation_id']
        with closing(_editable_database(changed)) as conn:
            raw = conn.execute('SELECT input_json FROM plan_step_runs WHERE id=?', (run_id,)).fetchone()[0]
            assert '-Infinity' in raw
            conn.execute('UPDATE plan_step_runs SET input_json=? WHERE id=?', (raw.replace('-Infinity', 'Infinity', 1), run_id))
            conn.commit()
        rejected = revalidate_recovery_archive(changed, expected_manifest_sha256=_rewrite_manifest(changed),
                                               frozen_binding=frozen)
        assert rejected['checks']['archive_integrity']['status'] == 'verified'
        assert rejected['checks']['original_process_fault_and_native_recovery']['status'] == 'mismatch'
