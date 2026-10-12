"""Saved report wording survives another draft without becoming metric evidence."""
import json

import pytest

from marvis.agent.service import fallback_word_conclusions
from marvis.api import _run_agent_word_conclusion_stage
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
            if delta:
                values.update(delta)
            return json.dumps(values)

    monkeypatch.setattr("marvis.agent.service._client", lambda _: Model())
    return prompts


@pytest.mark.parametrize("version", [1, 2])
def test_confirmed_report_remains_a_local_baseline_for_http_redraft(flow, monkeypatch, version):
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
    context = prompts[0]["report_revision_operation"]
    assert context["provenance"]["report_revision"] == revision
    assert "saved_report_narrative" not in prompts[0]["evidence"]
    assert saved["TEXT:model_scope"] not in json.dumps(prompts, ensure_ascii=False)
    assert "report_draft" not in prompts[0]["evidence"]
    assert result["message"]["metadata"]["report_revision"] == revision
    assert repo.get_report_values(task_id) == before


@pytest.mark.parametrize("version", [1, 2])
def test_saved_then_pending_local_edit_precedence_preserves_blanks(flow, monkeypatch, version):
    client, repo, task_id, _ = flow
    configure(repo, task_id, version)
    saved, revision = confirmed_report(repo, task_id)
    repo.add_agent_message(task_id, role="assistant", stage="word_conclusion_draft", content="编辑中",
                           metadata={"report_revision": revision, "draft_values": {
                               "TEXT:model_scope": "", "TEXT:sample_audience": "", "TEXT:model_overview": "本次明确改写",
                           }})
    prompts = model_with_optional_delta(monkeypatch)
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft", json={})
    assert response.status_code == 200, response.text
    values = response.json()["text_values"]
    assert values["TEXT:model_scope"] == ""
    assert values["TEXT:bad_sample_definition"] == saved["TEXT:bad_sample_definition"]
    assert values["TEXT:sample_audience"] == ""
    assert values["TEXT:model_overview"] == "本次明确改写"
    assert "本次明确改写" not in json.dumps(prompts, ensure_ascii=False)


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
    assert response.json()["message"]["metadata"]["confirmable"] is False
    assert repo.get_report_values(task_id)[0]["TEXT:model_scope"] == saved["TEXT:model_scope"]


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
    def forbidden_recalculation(**_kwargs):
        raise AssertionError("report revision cannot rerun scoring or metrics")
    for name in ("run_notebook_stage", "run_pmml_scoring_stage", "run_metrics_stage"):
        monkeypatch.setattr(f"marvis.agent.validation_app_service.{name}", forbidden_recalculation)
    assert _run_agent_word_conclusion_stage(
        repo, client.app.state.settings, task_id, {}, auto_accept=True,
        rewrite_instruction="按当前验证结果改写结论，其余不变。",
    )
    values, written_revision = repo.get_report_values(task_id)
    assert values["TEXT:bad_sample_definition"] == saved["TEXT:bad_sample_definition"]
    assert values["TEXT:model_scope"] == ""
    assert written_revision == revision + 1 and reports == [task_id]
    assert prompts[0]["evidence"]["validation_results"]["effectiveness"]["overall"] == [
        {"split": "oot", "ks": 0.31},
    ]
    assert measured == {"validation_results": {"effectiveness": {"overall": [{"split": "oot", "ks": 0.31}]}}}
    assert prompts[0]["report_revision_operation"]["provenance"]["report_revision"] == revision
    assert "saved_report_narrative" not in prompts[0]["evidence"]


@pytest.mark.parametrize("revision_kind", ["unsupported", "clear", "provider_failure"])
def test_async_revision_never_auto_accepts_fallback_or_loses_local_clears(flow, monkeypatch, revision_kind):
    from marvis.llm_client import LLMClientError

    client, repo, task_id, saved, revision, reports, measured = stage_setup(flow, monkeypatch)
    repo.add_agent_message(task_id, role="assistant", stage="word_conclusion_draft", content="编辑中",
                           metadata={"report_revision": revision, "draft_values": {"TEXT:model_scope": ""}})
    prompts = model_with_optional_delta(monkeypatch)
    if revision_kind == "provider_failure":
        class FailedModel:
            def complete(self, **_kwargs):
                raise LLMClientError("synthetic provider unavailable")
        monkeypatch.setattr("marvis.agent.service._client", lambda _: FailedModel())
    def forbidden_fallback(**_kwargs):
        raise AssertionError("rejected or cleared revision cannot use fallback")
    monkeypatch.setattr("marvis.agent.validation_app_service.fallback_word_conclusions", forbidden_fallback)
    before = repo.get_report_values(task_id)
    instruction = {
        "unsupported": "如果核验通过再改写结论，其余不变。",
        "clear": "删除结论，其余不变。",
        "provider_failure": "按当前验证结果改写结论，其余不变。",
    }[revision_kind]
    assert not _run_agent_word_conclusion_stage(
        repo, client.app.state.settings, task_id, {}, auto_accept=True, rewrite_instruction=instruction,
    )
    assert repo.get_report_values(task_id) == before and reports == []
    draft = next(message for message in reversed(repo.list_agent_messages(task_id))
                 if message["stage"] == "word_conclusion_draft")
    assert draft["metadata"]["draft_values"]["TEXT:model_scope"] == ""
    assert draft["metadata"]["draft_values"]["TEXT:bad_sample_definition"] == saved["TEXT:bad_sample_definition"]
    assert draft["metadata"]["confirmable"] is False
    if revision_kind == "clear":
        assert draft["metadata"]["draft_values"]["TEXT:final_validation_conclusion"] == ""
    if revision_kind == "unsupported":
        assert draft["metadata"]["outbound_context_status"] == "clarification_required"
        assert "需要澄清" in repo.list_agent_messages(task_id)[-1]["content"]
    assert prompts == []
    assert measured == {"validation_results": {"effectiveness": {"overall": [{"split": "oot", "ks": 0.31}]}}}


def test_explicit_local_business_revision_is_persisted_without_model_or_metric_run(flow, monkeypatch):
    client, repo, task_id, saved, revision, reports, measured = stage_setup(flow, monkeypatch)
    def forbidden_call(**_kwargs):
        raise AssertionError("local narrative edit cannot call a model or rerun metrics")
    for name in ("run_notebook_stage", "run_pmml_scoring_stage", "run_metrics_stage", "fallback_word_conclusions"):
        monkeypatch.setattr(f"marvis.agent.validation_app_service.{name}", forbidden_call)
    monkeypatch.setattr("marvis.agent.service._client", forbidden_call)
    assert _run_agent_word_conclusion_stage(
        repo, client.app.state.settings, task_id, {}, auto_accept=True,
        rewrite_instruction="坏样本定义改成 MOB6 逾期 >= 30 天，其他不变。",
    )
    values, written_revision = repo.get_report_values(task_id)
    assert values["TEXT:bad_sample_definition"] == "MOB6 逾期 >= 30 天"
    assert values["TEXT:model_scope"] == saved["TEXT:model_scope"]
    assert values["TEXT:final_validation_conclusion"] == saved["TEXT:final_validation_conclusion"]
    assert written_revision == revision + 1 and reports == [task_id]
    generated = next(message for message in reversed(repo.list_agent_messages(task_id))
                     if message["stage"] == "word_conclusion_generated")
    assert generated["metadata"]["narrative_source"] == "local_edited"
    assert measured == {"validation_results": {"effectiveness": {"overall": [{"split": "oot", "ks": 0.31}]}}}


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
            rewrite_instruction="按当前验证结果改写结论，其余不变。",
        )
    assert repo.get_report_values(task_id)[0]["TEXT:model_scope"] == "生成期间保存的新范围"
    assert repo.get_report_values(task_id)[1] == revision + 1
    assert reports == []
    draft = next(m for m in reversed(repo.list_agent_messages(task_id)) if m["stage"] == "word_conclusion_draft")
    assert draft["metadata"]["report_revision"] == revision
    assert prompts[0]["report_revision_operation"]["provenance"]["report_revision"] == revision
