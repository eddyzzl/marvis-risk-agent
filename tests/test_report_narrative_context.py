"""Saved report wording survives another draft without becoming metric evidence."""
import json

import pytest

from marvis.agent.service import fallback_word_conclusions
from marvis.api import _run_agent_word_conclusion_stage
from marvis.repositories.tasks import AGENT_REPORT_WRITABLE_KEYS
from marvis.state_machine import ConflictError
from tests import test_report_draft_generation_race as draft_fixture
from tests.test_report_draft_generation_race import VALUES, configure


@pytest.fixture
def flow(tmp_path, monkeypatch):
    return draft_fixture.flow.__wrapped__(tmp_path, monkeypatch)


def confirmed_report(repo, task_id):
    saved = {
        **VALUES,
        "TEXT:model_scope": f"原始业务范围 {task_id}",
        "TEXT:bad_sample_definition": "MOB3 DPD60，按既定业务定义",
        "TEXT:good_sample_definition": "",
    }
    revision = repo.update_agent_report_conclusions_with_audit(
        task_id, saved, 0,
        audit={"kind": "report.agent_conclusions.confirm", "target_ref": task_id,
               "outcome": "succeeded", "detail": {"keys": sorted(saved)}},
    )
    repo.add_agent_message(task_id, role="assistant", stage="word_conclusion_confirmed",
                           content="已确认", metadata={"revision": revision})
    return saved, revision


def model_with_optional_delta(monkeypatch, *, delta=None, hook=None):
    prompts = []
    fired = False

    class Model:
        def complete(self, **kwargs):
            nonlocal fired
            prompts.append(json.loads(kwargs["user_prompt"]))
            if hook and not fired:
                fired = True
                hook()
            schema = kwargs.get("json_schema", {}).get("schema", {})
            fields = schema.get("required", [key for key in VALUES if key != "TEXT:model_scope"])
            values = {key: VALUES[key] for key in fields}
            if delta and (not schema or "TEXT:model_training_description" in fields):
                values.update(delta)
            return json.dumps(values)

    monkeypatch.setattr("marvis.agent.service._client", lambda _: Model())
    return prompts


@pytest.mark.parametrize("version", [1, 2])
def test_confirmed_report_is_a_raw_narrative_baseline_for_http_redraft(flow, monkeypatch, version):
    client, repo, task_id, _ = flow
    configure(repo, task_id, version)
    saved, revision = confirmed_report(repo, task_id)
    before = repo.get_report_values(task_id)
    prompts = model_with_optional_delta(monkeypatch)
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 200, response.text
    result = response.json()
    for key in ("TEXT:model_scope", "TEXT:bad_sample_definition", "TEXT:good_sample_definition"):
        assert result["text_values"][key] == saved[key]
    context = prompts[0]["evidence"]["saved_report_narrative"]
    assert context["report_revision"] == revision
    assert set(context["text_values"]) <= AGENT_REPORT_WRITABLE_KEYS
    assert "TEXT:report_title" not in context["text_values"]
    assert "report_draft" not in prompts[0]["evidence"]
    assert result["message"]["metadata"]["report_revision"] == revision
    assert repo.get_report_values(task_id) == before


@pytest.mark.parametrize("version", [1, 2])
def test_saved_then_pending_then_explicit_delta_precedence_preserves_blanks(flow, monkeypatch, version):
    client, repo, task_id, _ = flow
    configure(repo, task_id, version)
    saved, revision = confirmed_report(repo, task_id)
    repo.add_agent_message(task_id, role="assistant", stage="word_conclusion_draft", content="编辑中",
                           metadata={"report_revision": revision, "draft_values": {
                               "TEXT:model_scope": "", "TEXT:sample_audience": "待改写客群",
                           }})
    model_with_optional_delta(monkeypatch, delta={"TEXT:sample_audience": "", "TEXT:model_overview": "本次明确改写"})
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 200, response.text
    values = response.json()["text_values"]
    assert values["TEXT:model_scope"] == ""
    assert values["TEXT:bad_sample_definition"] == saved["TEXT:bad_sample_definition"]
    assert values["TEXT:sample_audience"] == ""
    assert values["TEXT:model_overview"] == "本次明确改写"


def test_fallback_keeps_saved_business_text_without_using_it_as_measured_evidence(flow):
    _, repo, task_id, _ = flow
    saved, revision = confirmed_report(repo, task_id)
    values = fallback_word_conclusions(task=repo.get_task(task_id), evidence={
        "saved_report_narrative": {"report_revision": revision, "text_values": saved},
        "report_draft": {"text_values": {"TEXT:model_scope": ""}},
    })
    assert values["TEXT:model_scope"] == ""
    assert values["TEXT:bad_sample_definition"] == saved["TEXT:bad_sample_definition"]
    assert "缺少足以" in values["TEXT:final_validation_conclusion"]


@pytest.mark.parametrize("empty_nontext", [None, False, 0])
def test_v1_nontext_falsy_response_is_not_an_explicit_narrative_clear(flow, monkeypatch, empty_nontext):
    client, repo, task_id, _ = flow
    configure(repo, task_id, 1)
    saved, _ = confirmed_report(repo, task_id)
    model_with_optional_delta(monkeypatch, delta={"TEXT:model_scope": empty_nontext})
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 200, response.text
    assert response.json()["text_values"]["TEXT:model_scope"] == saved["TEXT:model_scope"]


def stage_setup(flow, monkeypatch):
    client, repo, task_id, _ = flow
    configure(repo, task_id, 1)
    saved, revision = confirmed_report(repo, task_id)
    reports = []
    measured = {"validation_results": {"effectiveness": {"overall": [{"split": "oot", "ks": 0.31}]}}}
    monkeypatch.setattr("marvis.agent.validation_app_service.agent_evidence_from_settings_impl",
                        lambda *_: measured)
    monkeypatch.setattr("marvis.agent.validation_app_service.run_report_stage",
                        lambda **kwargs: reports.append(kwargs["task_id"]))
    monkeypatch.setattr("marvis.agent.validation_app_service.agent_pipeline_settings", lambda *_: object())
    return client, repo, task_id, saved, revision, reports, measured


def test_async_redraft_preserves_saved_text_and_persists_pending_explicit_clear(flow, monkeypatch):
    client, repo, task_id, saved, revision, reports, measured = stage_setup(flow, monkeypatch)
    repo.add_agent_message(task_id, role="assistant", stage="word_conclusion_draft", content="编辑中",
                           metadata={"report_revision": revision, "draft_values": {"TEXT:model_scope": ""}})
    prompts = model_with_optional_delta(monkeypatch)
    assert _run_agent_word_conclusion_stage(
        repo, client.app.state.settings, task_id, {}, auto_accept=True,
        rewrite_instruction="仅调整最终结论，其他业务定义不变，范围保持清空。",
    )
    values, written_revision = repo.get_report_values(task_id)
    assert values["TEXT:bad_sample_definition"] == saved["TEXT:bad_sample_definition"]
    assert values["TEXT:model_scope"] == ""
    assert written_revision == revision + 1 and reports == [task_id]
    assert prompts[0]["evidence"]["validation_results"]["effectiveness"]["overall"] == [
        {"split": "oot", "ks": 0.31},
    ]
    assert measured == {"validation_results": {"effectiveness": {"overall": [{"split": "oot", "ks": 0.31}]}}}
    assert prompts[0]["evidence"]["saved_report_narrative"]["report_revision"] == revision


@pytest.mark.parametrize("timing", ["before_model", "during_model"])
def test_async_generation_never_uses_a_new_report_revision_to_approve_old_text(flow, monkeypatch, timing):
    client, repo, task_id, _, revision, reports, _ = stage_setup(flow, monkeypatch)

    def edit():
        response = client.put(f"/api/tasks/{task_id}/report-fields", headers={"If-Match": str(revision)},
                              json={"text_values": {"TEXT:model_scope": "生成期间保存的新范围"}})
        assert response.status_code == 200, response.text

    prompts = model_with_optional_delta(monkeypatch, hook=edit if timing == "during_model" else None)
    if timing == "before_model":
        def memory(*_args, **_kwargs):
            edit()
            return {}
        monkeypatch.setattr("marvis.agent.validation_stages.agent_memory_context_from_store", memory)
    with pytest.raises(ConflictError):
        _run_agent_word_conclusion_stage(
            repo, client.app.state.settings, task_id, {}, auto_accept=True,
            rewrite_instruction="只调整结论语气",
        )
    assert repo.get_report_values(task_id)[0]["TEXT:model_scope"] == "生成期间保存的新范围"
    assert repo.get_report_values(task_id)[1] == revision + 1
    assert reports == []
    draft = next(m for m in reversed(repo.list_agent_messages(task_id)) if m["stage"] == "word_conclusion_draft")
    assert draft["metadata"]["report_revision"] == revision
    assert prompts[0]["evidence"]["saved_report_narrative"]["report_revision"] == revision
