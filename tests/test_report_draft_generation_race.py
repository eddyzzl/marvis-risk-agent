"""Publish a generated report draft only against the state it actually read."""
import json

import pytest
from fastapi.testclient import TestClient

from marvis.app import create_app
from marvis.db import TaskRepository
from marvis.db_schema import connect
from marvis.routers import validation_agent


VALUES = {
    "TEXT:pressure_test_summary": "已核对压力测试证据。",
    "TEXT:pressure_impact_recommendation": "持续关注外部数据可用性。",
    "TEXT:final_validation_conclusion": "当前证据仅支持技术验证。",
    "TEXT:model_training_description": "本模型采用逻辑回归。",
    "TEXT:model_scope": "仅限既有客户。",
}


@pytest.fixture
def flow(tmp_path, monkeypatch):
    client = TestClient(create_app(tmp_path))
    task = client.post("/api/tasks", json={
        "model_name": "审查T卡", "validator": "qa", "source_dir": str(tmp_path),
        "run_mode": "agent",
    }).json()
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    control = {"hook": None, "calls": 0, "prompts": []}

    class Model:
        def complete(self, **kwargs):
            control["calls"] += 1
            control["prompts"].append(json.loads(kwargs["user_prompt"]))
            hook, control["hook"] = control["hook"], None
            if hook is not None:
                hook()
            fields = kwargs.get("json_schema", {}).get("schema", {}).get("properties", VALUES)
            return json.dumps({key: value for key, value in VALUES.items() if key in fields})

    monkeypatch.setattr(validation_agent, "resolve_agent_model", lambda *_: {})
    monkeypatch.setattr("marvis.agent.service._client", lambda _: Model())
    return client, repo, task["id"], control


def configure(repo, task_id, version):
    with connect(repo.db_path) as conn:
        conn.execute(
            "UPDATE tasks SET validation_workflow_version=?, status='writing_artifacts' WHERE id=?",
            (version, task_id),
        )


def saved_draft(repo, task_id):
    return repo.add_agent_message(
        task_id, role="assistant", stage="word_conclusion_draft", content="现有草稿",
        metadata={"draft_values": VALUES, "report_revision": 0, "draft_edit_revision": 0},
    )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("has_draft", [False, True])
def test_unchanged_snapshot_publishes_initial_or_revised_draft(flow, version, has_draft):
    client, repo, task_id, control = flow
    configure(repo, task_id, version)
    original = saved_draft(repo, task_id) if has_draft else None
    before = repo.get_report_values(task_id)
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 200, response.text
    message = response.json()["message"]
    assert message["metadata"]["report_revision"] == before[1]
    assert message["metadata"]["fallback"] is False
    assert repo.get_report_values(task_id) == before
    assert repo.get_active_job_kind(task_id) is None
    assert control["calls"] == (4 if version == 2 else 1)
    if original:
        assert control["prompts"][0]["evidence"]["report_draft"]["message_id"] == original["id"]
        assert message["id"] != original["id"]


@pytest.mark.parametrize("version", [1, 2])
def test_save_during_generation_preserves_new_edit_and_rejects_stale_publication(flow, version):
    client, repo, task_id, control = flow
    configure(repo, task_id, version)
    original = saved_draft(repo, task_id)

    def save():
        response = client.put(f"/api/tasks/{task_id}/agent/report-draft", json={
            "revision": 0, "draft_message_id": original["id"], "draft_edit_revision": 0,
            "text_values": {"TEXT:model_scope": "另一窗口刚保存的新范围"},
        })
        assert response.status_code == 200, response.text

    control["hook"] = save
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 409, response.text
    messages = repo.list_agent_messages(task_id)
    assert len(messages) == 1
    assert messages[0]["metadata"]["draft_values"]["TEXT:model_scope"] == "另一窗口刚保存的新范围"
    assert messages[0]["metadata"]["draft_edit_revision"] == 1


@pytest.mark.parametrize("has_draft", [False, True])
def test_concurrent_generation_has_only_one_winner_even_without_prior_draft(flow, has_draft):
    client, repo, task_id, control = flow
    configure(repo, task_id, 2)
    if has_draft:
        saved_draft(repo, task_id)
    winner = []

    def generate_again():
        response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
        assert response.status_code == 200, response.text
        winner.append(response.json()["message"]["id"])

    control["hook"] = generate_again
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 409, response.text
    messages = repo.list_agent_messages(task_id)
    assert len(messages) == (2 if has_draft else 1)
    assert messages[-1]["id"] == winner[0]


def test_confirmation_during_generation_is_not_superseded(flow):
    client, repo, task_id, control = flow
    configure(repo, task_id, 1)
    original = saved_draft(repo, task_id)

    def confirm():
        [confirmation] = repo.confirm_agent_report_batch([{
            "task_id": task_id, "text_values": VALUES, "expected_revision": 0,
            "draft_message_id": original["id"], "draft_edit_revision": 0,
        }])
        # Even a job which has already finished must invalidate the snapshot.
        repo.finish_job(confirmation["job_id"], status="succeeded")

    control["hook"] = confirm
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 409, response.text
    assert repo.list_agent_messages(task_id)[-1]["stage"] == "word_conclusion_confirmed"
    assert repo.get_report_values(task_id)[1] == 1


@pytest.mark.parametrize("finished", [False, True])
def test_job_started_during_generation_invalidates_snapshot_even_if_finished(flow, finished):
    client, repo, task_id, control = flow
    configure(repo, task_id, 2)

    def start_job():
        job_id = repo.start_job(task_id, "metrics")
        if finished:
            repo.finish_job(job_id, status="succeeded")

    control["hook"] = start_job
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 409, response.text
    assert repo.list_agent_messages(task_id) == []


def test_active_job_at_start_rejects_before_any_model_call(flow):
    client, repo, task_id, control = flow
    repo.start_job(task_id, "metrics")
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 409, response.text
    assert control["calls"] == 0
    assert repo.list_agent_messages(task_id) == []


def test_report_revision_changed_during_generation_is_not_attached_to_old_text(flow):
    client, repo, task_id, control = flow
    control["hook"] = lambda: repo.update_agent_report_conclusions(task_id, VALUES, 0)
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 409, response.text
    assert repo.get_report_values(task_id)[1] == 1
    assert repo.list_agent_messages(task_id) == []


def test_abandoned_placeholder_does_not_hide_the_saved_draft_context(flow):
    client, repo, task_id, control = flow
    original = saved_draft(repo, task_id)
    repo.add_agent_message(task_id, role="assistant", stage="word_conclusion_draft",
                           content="", metadata={"streaming": True})
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 200, response.text
    assert control["prompts"][0]["evidence"]["report_draft"]["message_id"] == original["id"]
    assert response.json()["message"]["metadata"]["generation_source"] == {
        "message_id": original["id"], "draft_edit_revision": 0, "report_revision": 0,
    }
