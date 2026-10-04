"""Selection references survive real training and cannot certify inner CV."""

import pytest

from marvis.plugins.manifest import ToolRef
from marvis.repositories.modeling import ModelingRepository
from marvis.repositories.task_artifacts import TaskArtifactRepository
from tests.test_selection_evidence import _source, _run


def selected(tmp_path):
    runner, registry, repo, dataset = _source(tmp_path)
    result = _run(runner, dataset, 'select_features', iv_min=0.0)
    assert result.ok, result.error
    return runner, registry, repo, dataset, result.output


def inputs(dataset, selection, tool='train_model'):
    payload = {'dataset_id': dataset.id, 'features': selection['selected'], 'target_col': 'y',
               'split_col': 'split', 'split_values': {'train': 'train', 'test': 'test', 'oot': 'oot'},
               'seed': 3, 'selection_evidence_refs': [selection['selection_evidence_ref']]}
    if tool == 'train_models':
        payload['recipes'] = ['lr']
    else:
        payload['recipe'] = 'lr'
    if tool == 'tune_hyperparameters':
        payload.update(n_trials=1, cv_folds=3)
    return payload


@pytest.mark.parametrize('tool', ['train_model', 'train_models', 'tune_hyperparameters'])
def test_training_and_cache_revalidate_selection_receipt(tmp_path, tool):
    runner, registry, repo, dataset, selection = selected(tmp_path)
    payload = inputs(dataset, selection, tool)
    first = runner.invoke(ToolRef('modeling', tool), payload, task_id='task-feature')
    assert first.ok, first.error
    if tool != 'tune_hyperparameters':
        experiment_id = first.output['experiment_id'] if tool == 'train_model' else first.output['best_experiment_id']
        from marvis.packs.modeling.experiment import ExperimentStore
        experiment = ExperimentStore(repo.db_path).get(experiment_id)
        artifact = ModelingRepository(repo.db_path).get_model_artifact(experiment.artifact_id)
        evidence = artifact.params['selection_evidence']
        assert evidence['references'] == payload['selection_evidence_refs']
        assert evidence['core_evidence_assurance'] == 'recorded'
        assert evidence['assurance'] == 'unknown'
        assert evidence['inner_validation']['assurance'] == 'not_established'
    ref = selection['selection_evidence_ref']
    record = TaskArtifactRepository(repo.db_path).get_for_task('task-feature', ref['artifact_id'])
    (registry.datasets_root.parent / record['path']).write_bytes(b'changed selection receipt')
    repeated = runner.invoke(ToolRef('modeling', tool), payload, task_id='task-feature')
    assert not repeated.ok
    assert 'selection evidence content changed' in str(repeated.error)


def test_params_cannot_supply_platform_selection_claim(tmp_path):
    runner, _, repo, dataset, selection = selected(tmp_path)
    payload = inputs(dataset, selection)
    del payload['selection_evidence_refs']
    payload['params'] = {'selection_evidence': {'assurance': 'verified'},
                         'selection_evidence_refs': [selection['selection_evidence_ref']]}
    result = runner.invoke(ToolRef('modeling', 'train_model'), payload, task_id='task-feature')
    assert result.ok, result.error
    evidence = ModelingRepository(repo.db_path).get_model_artifact(result.output['artifact_id']).params['selection_evidence']
    assert evidence['core_evidence_assurance'] == evidence['assurance'] == 'unknown'
    assert evidence['references'] == []


def test_wrong_source_reference_rejected_even_beside_valid_reference(tmp_path):
    runner, registry, _, dataset, selection = selected(tmp_path)
    source_path = tmp_path / 'other.csv'
    frame = registry.read_authenticated_parquet_snapshot(dataset.id)
    frame['x'] += 1
    frame.to_csv(source_path, index=False)
    other = registry.register_from_upload('task-feature', source_path, role='sample')
    wrong = _run(runner, other, 'select_features', iv_min=0.0)
    assert wrong.ok, wrong.error
    payload = inputs(dataset, selection)
    payload['selection_evidence_refs'].append(wrong.output['selection_evidence_ref'])
    result = runner.invoke(ToolRef('modeling', 'train_model'), payload, task_id='task-feature')
    assert not result.ok
    assert 'authenticated ancestor' in str(result.error)


@pytest.mark.parametrize('invalid', ['artifact_id', 'content_hash', 'task'])
def test_native_training_rejects_selection_identity_mismatch(tmp_path, invalid):
    runner, registry, repo, dataset, selection = selected(tmp_path)
    payload = inputs(dataset, selection)
    if invalid == 'task':
        from marvis.db import TaskRepository
        from marvis.domain import TaskCreate

        task = TaskRepository(repo.db_path).create_task(TaskCreate(
            model_name='Foreign selection', model_version='v1', validator='test',
            source_dir=str(tmp_path), task_type='feature_analysis',
        ))
        source = tmp_path / 'foreign.csv'
        registry.read_authenticated_parquet_snapshot(dataset.id).to_csv(source, index=False)
        foreign = registry.register_from_upload(task.id, source, role='sample')
        result = runner.invoke(ToolRef('modeling', 'select_features'), {
            'dataset_id': foreign.id, 'features': ['x', 'z'], 'target_col': 'y',
            'split_col': 'split', 'iv_min': 0.0, 'seed': 3,
        }, task_id=task.id)
        assert result.ok, result.error
        payload['selection_evidence_refs'] = [result.output['selection_evidence_ref']]
    else:
        payload['selection_evidence_refs'] = [{
            **selection['selection_evidence_ref'],
            invalid: '0' * 64 if invalid == 'content_hash' else 'nonexistent-selection',
        }]
    result = runner.invoke(ToolRef('modeling', 'train_model'), payload, task_id='task-feature')
    assert not result.ok
    assert 'selection evidence identity mismatch' in str(result.error)


def test_changed_selection_stops_refit_and_model_card_delivery(tmp_path):
    runner, registry, repo, dataset, selection = selected(tmp_path)
    trained = runner.invoke(ToolRef('modeling', 'train_model'), inputs(dataset, selection), task_id='task-feature')
    assert trained.ok, trained.error
    record = TaskArtifactRepository(repo.db_path).get_for_task(
        'task-feature', selection['selection_evidence_ref']['artifact_id'],
    )
    (registry.datasets_root.parent / record['path']).write_bytes(b'changed before delivery')
    refit = runner.invoke(ToolRef('modeling', 'select_experiment'), {
        'experiment_ids': [trained.output['experiment_id']], 'refit_on_train_plus_test': True,
    }, task_id='task-feature')
    assert refit.ok, refit.error
    assert not refit.output['refit']['applied']
    assert refit.output['artifact_id'] == trained.output['artifact_id']
    assert 'selection evidence content changed' in refit.output['refit']['reason']
    delivered = runner.invoke(ToolRef('modeling', 'post_training_action'), {
        'experiment_id': trained.output['experiment_id'],
        'sample_dataset_id': dataset.id, 'actions': ['export_pmml'],
    }, task_id='task-feature')
    assert not delivered.ok
    assert 'selection evidence content changed' in str(delivered.error)


def test_native_mask_chain_preserves_both_screen_and_refinement_sources(tmp_path):
    runner, registry, repo, dataset, selection = selected(tmp_path)
    column = selection['selected'][0]
    value = float(registry.read_authenticated_parquet_snapshot(dataset.id)[column].iloc[0])
    masked = runner.invoke(ToolRef('modeling', 'resolve_special_values'), {
        'dataset_id': dataset.id, 'features': selection['selected'],
        'sentinel_columns': {column: [[value, 1 / 120]]},
        'decisions': {column: {'action': 'mask'}}, 'seed': 3,
    }, task_id='task-feature')
    assert masked.ok, masked.error
    derived = registry.get(masked.output['result_dataset_id'])
    refined = _run(runner, derived, 'select_features', iv_min=0.0)
    assert refined.ok, refined.error
    payload = inputs(derived, refined.output)
    payload['selection_evidence_refs'].insert(0, selection['selection_evidence_ref'])
    payload['special_value_governance'] = masked.output['governance']
    trained = runner.invoke(ToolRef('modeling', 'train_model'), payload, task_id='task-feature')
    assert trained.ok, trained.error
    evidence = ModelingRepository(repo.db_path).get_model_artifact(trained.output['artifact_id']).params['selection_evidence']
    assert len(evidence['sources']) == 2
    assert evidence['current_row_mapping'] == 'recorded'
    assert all(source['auxiliary_diagnostics']['assurance'] == 'unknown' for source in evidence['sources'])
    assert evidence['sources'][0]['current_memberships']['fit']['rows'] == 40
    assert evidence['sources'][0]['mapping_artifact_ids']


def test_refit_and_model_card_keep_conditional_validation_boundary(tmp_path):
    runner, _, repo, dataset, selection = selected(tmp_path)
    trained = runner.invoke(ToolRef('modeling', 'train_model'), inputs(dataset, selection), task_id='task-feature')
    assert trained.ok, trained.error
    refit = runner.invoke(ToolRef('modeling', 'select_experiment'), {
        'experiment_ids': [trained.output['experiment_id']], 'refit_on_train_plus_test': True,
    }, task_id='task-feature')
    assert refit.ok, refit.error
    assert refit.output['refit']['applied']
    artifact = ModelingRepository(repo.db_path).get_model_artifact(refit.output['artifact_id'])
    assert artifact.params['selection_evidence']['references'] == [selection['selection_evidence_ref']]
    delivered = runner.invoke(ToolRef('modeling', 'post_training_action'), {
        'experiment_id': refit.output['selected_experiment_id'],
        'sample_dataset_id': dataset.id, 'actions': ['export_pmml'],
    }, task_id='task-feature')
    assert delivered.ok, delivered.error
    card = delivered.output['model_card']
    evidence = card['training']['selection_evidence']
    assert evidence['core_evidence_assurance'] == 'recorded'
    assert evidence['inner_validation'] == {'assurance': 'not_established', 'mode': 'conditional_on_outer_selection'}
    from pathlib import Path
    markdown = Path(delivered.output['model_card_markdown_path']).read_text()
    assert '完整独立性未建立' in markdown


def test_governed_native_training_binds_selection_to_evidence_and_rejects_tampering(tmp_path):
    from marvis.packs.modeling import feature_tools
    from tests.test_modeling_training_evidence_tool import _fixture, _run as run_governed, _binding

    fx = _fixture(tmp_path)
    dataset = fx['dataset']
    selection = feature_tools.tool_screen_features({
        'dataset_id': dataset.id, 'features': ['x1', 'x2'], 'target_col': 'bad',
        'split_col': 'model_split', 'leakage_ks': 1.0,
    }, fx['ctx'])
    fx['inputs']['selection_evidence_refs'] = [selection['selection_evidence_ref']]
    output = run_governed(fx)
    artifact = ModelingRepository(fx['settings'].db_path).get_model_artifact(output['model_artifact_id'])
    assert artifact.params['selection_evidence']['core_evidence_assurance'] == 'recorded'
    assert _binding(fx, output).evidence['evidence_id'] == output['evidence_id']
    ref = selection['selection_evidence_ref']
    record = TaskArtifactRepository(fx['settings'].db_path).get_for_task(fx['task'].id, ref['artifact_id'])
    (fx['runtime'].registry.datasets_root.parent / record['path']).write_bytes(b'corrupt selection')
    with pytest.raises(Exception, match='selection evidence content changed'):
        run_governed(fx)
