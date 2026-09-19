from concurrent.futures import ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.db_schema import connect
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


def _complete_draft(repo, task_id, *, scope="旧范围"):
    values = {
        "TEXT:pressure_test_summary": "压力测试草稿",
        "TEXT:pressure_impact_recommendation": "风险建议草稿",
        "TEXT:final_validation_conclusion": "原结论",
        "TEXT:model_scope": scope,
    }
    return repo.add_agent_message(
        task_id, role="assistant", stage="word_conclusion_draft",
        content="原结论", metadata={"draft_values": values, "report_revision": 0},
    )


def test_saved_narrative_reaches_chat_and_rewrite_prompts_without_changing_metrics(draft):
    from marvis.agent.service import _chat_prompt, _stage_prompt
    from marvis.agent.validation_app_service import agent_chat_evidence
    from marvis.agent.validation_stages import _word_conclusion_evidence_with_stage_summaries

    client, repo, task_id, _ = draft
    message = _complete_draft(repo, task_id)
    repo.save_agent_report_draft(
        task_id, message_id=message["id"], edit_revision=0, expected_revision=0,
        values={"TEXT:final_validation_conclusion": "用户保存的新结论", "TEXT:model_scope": ""},
    )
    original_report = repo.get_report_values(task_id)
    task = repo.get_task(task_id)
    request = SimpleNamespace(app=client.app)
    conversation = repo.list_agent_messages(task_id)
    chat_evidence = agent_chat_evidence(request, repo, task, conversation)
    chat_prompt = json.loads(_chat_prompt(
        task=task, user_message="解释我保存的结论", conversation=conversation,
        evidence=chat_evidence,
    ))
    assert "用户保存的新结论" in json.dumps(chat_prompt, ensure_ascii=False)
    assert chat_evidence["report_draft"]["text_values"]["TEXT:model_scope"] == ""
    assert chat_evidence["report_fields"]["text_values"] != chat_evidence["report_draft"]["text_values"]

    # The asynchronous rewrite creates a placeholder before gathering evidence.
    repo.add_agent_message(task_id, role="assistant", stage="word_conclusion_draft",
                           content="", metadata={"streaming": True})
    deterministic = {"validation_results": {"basic_info": {"n_samples": 17}}}
    rewrite_evidence = _word_conclusion_evidence_with_stage_summaries(repo, task_id, deterministic)
    rewrite_prompt = json.loads(_stage_prompt(
        task=task, stage="word_conclusion_draft", evidence=rewrite_evidence,
        user_instruction="只调整最终结论的语气",
    ))
    assert rewrite_prompt["evidence"]["report_draft"]["text_values"]["TEXT:final_validation_conclusion"] == "用户保存的新结论"
    assert rewrite_prompt["evidence"]["report_draft"]["text_values"]["TEXT:model_scope"] == ""
    assert rewrite_evidence["validation_results"] == deterministic["validation_results"]
    assert repo.get_report_values(task_id) == original_report


def test_batch_confirmation_preserves_saved_empty_narrative(draft):
    from marvis.agent.validation_app_service import latest_pending_agent_report_draft

    _, repo, task_id, _ = draft
    with connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE tasks SET validation_workflow_version=1, status='writing_artifacts', "
            "report_values_json=? WHERE id=?",
            (json.dumps({"TEXT:model_scope": "先前写入的旧范围"}), task_id),
        )
    message = _complete_draft(repo, task_id)
    repo.save_agent_report_draft(
        task_id, message_id=message["id"], edit_revision=0, expected_revision=0,
        values={"TEXT:model_scope": ""},
    )
    pending = latest_pending_agent_report_draft(repo.list_agent_messages(task_id))
    assert pending["values"]["TEXT:model_scope"] == ""
    repo.confirm_agent_report_batch([{
        "task_id": task_id, "text_values": pending["values"],
        "expected_revision": pending["report_revision"],
        "draft_message_id": pending["message_id"],
        "draft_edit_revision": pending["draft_edit_revision"],
    }])
    assert repo.get_report_values(task_id)[0]["TEXT:model_scope"] == ""


def test_incomplete_saved_draft_does_not_select_older_confirmable_draft(draft):
    from marvis.agent.validation_app_service import latest_pending_agent_report_draft

    _, repo, task_id, _ = draft
    _complete_draft(repo, task_id)
    current = _complete_draft(repo, task_id)
    repo.save_agent_report_draft(
        task_id, message_id=current["id"], edit_revision=0, expected_revision=0,
        values={"TEXT:final_validation_conclusion": ""},
    )
    assert latest_pending_agent_report_draft(repo.list_agent_messages(task_id)) == {}


@pytest.mark.parametrize("semantic", [False, True])
def test_chat_confirmation_rejects_concurrent_saved_revision(draft, monkeypatch, semantic):
    from marvis.routers import validation_agent

    client, repo, task_id, _ = draft
    with connect(repo.db_path) as conn:
        conn.execute("UPDATE tasks SET validation_workflow_version=1, status='writing_artifacts' WHERE id=?", (task_id,))
    message = _complete_draft(repo, task_id)
    before = repo.get_report_values(task_id)

    def concurrent_save(*_args, **_kwargs):
        repo.save_agent_report_draft(
            task_id, message_id=message["id"], edit_revision=0, expected_revision=0,
            values={"TEXT:final_validation_conclusion": "另一个窗口的新结论"},
        )

    monkeypatch.setattr(validation_agent, "capture_user_preference_memory", concurrent_save)
    monkeypatch.setattr(validation_agent, "resolve_agent_model", lambda *_args: {})
    monkeypatch.setattr(validation_agent, "review_validation_semantic_authorization", lambda *_args, **_kwargs: {"authorized": True})
    response = client.post(f"/api/tasks/{task_id}/agent/messages", json={
        "content": "可以定稿了" if semantic else "确认",
    })
    assert response.status_code == 409, response.text
    assert repo.get_report_values(task_id) == before
    assert repo.get_active_job_kind(task_id) is None
    assert not any(m["stage"] == "word_conclusion_confirmed" for m in repo.list_agent_messages(task_id))
