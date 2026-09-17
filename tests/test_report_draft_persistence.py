from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.state_machine import ConflictError


@pytest.fixture
def draft(tmp_path):
    client = TestClient(create_app(tmp_path))
    task = client.post('/api/tasks', json={
        'model_name': '演示T卡', 'validator': 'qa', 'source_dir': str(tmp_path), 'run_mode': 'agent',
    }).json()
    repo = TaskRepository(tmp_path / 'marvis.sqlite')
    values = {'TEXT:final_validation_conclusion': '待核对结论'}
    message = repo.add_agent_message(task['id'], role='assistant', stage='word_conclusion_draft',
                                     content='演示草稿', metadata={'draft_values': values, 'report_revision': 0})
    return client, repo, task['id'], message


def test_save_survives_new_repository_without_approval_metrics_or_job(draft):
    client, repo, task_id, message = draft
    before = repo.get_report_values(task_id)
    body = {'revision': 0, 'draft_message_id': message['id'], 'draft_edit_revision': 0,
            'text_values': {'TEXT:final_validation_conclusion': '用户修改', 'TEXT:model_scope': ''}}
    response = client.put(f'/api/tasks/{task_id}/agent/report-draft', json=body)
    assert response.status_code == 200, response.text
    saved = TaskRepository(repo.db_path).get_agent_message(task_id, message['id'])
    assert saved['metadata']['draft_values']['TEXT:final_validation_conclusion'] == '用户修改'
    assert saved['metadata']['draft_edit_revision'] == 1
    assert repo.get_report_values(task_id) == before
    assert repo.get_active_job_kind(task_id) is None
    assert [m['stage'] for m in repo.list_agent_messages(task_id)] == ['word_conclusion_draft']
    assert client.put(f'/api/tasks/{task_id}/agent/report-draft', json=body).status_code == 409


def test_concurrent_saves_reject_lost_update(draft):
    _, repo, task_id, message = draft
    def save(value):
        try:
            repo.save_agent_report_draft(task_id, message_id=message['id'], edit_revision=0,
                                        expected_revision=0, values={'TEXT:final_validation_conclusion': value})
            return 'saved'
        except ConflictError:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(save, ['first', 'second'])) == ['conflict', 'saved']


def test_save_rejects_superseded_draft_and_computed_fields(draft):
    client, repo, task_id, message = draft
    body = {'revision': 0, 'draft_message_id': message['id'], 'draft_edit_revision': 0,
            'text_values': {'TEXT:KS': '0.99'}}
    assert client.put(f'/api/tasks/{task_id}/agent/report-draft', json=body).status_code == 422
    repo.add_agent_message(task_id, role='assistant', stage='word_conclusion_confirmed', content='已确认')
    body['text_values'] = {'TEXT:final_validation_conclusion': '过期修改'}
    assert client.put(f'/api/tasks/{task_id}/agent/report-draft', json=body).status_code == 409
    assert repo.get_agent_message(task_id, message['id'])['metadata']['draft_values']['TEXT:final_validation_conclusion'] == '待核对结论'


def test_confirmation_version_check_is_atomic_and_preserves_report_values(draft):
    _, repo, task_id, message = draft
    repo.save_agent_report_draft(task_id, message_id=message['id'], edit_revision=0,
                                expected_revision=0, values={'TEXT:final_validation_conclusion': '新版'})
    before = repo.get_report_values(task_id)
    with pytest.raises(ConflictError):
        repo.update_agent_report_conclusions_with_audit(
            task_id, {'TEXT:final_validation_conclusion': '旧版'}, 0,
            draft_message_id=message['id'], draft_edit_revision=0,
            audit={'kind': 'test', 'target_ref': task_id},
        )
    assert repo.get_report_values(task_id) == before
