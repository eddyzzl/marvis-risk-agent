"""Regressions from the 2026-10-19 hands-on Agent eval (model validation, V2).

Each test pins one observed failure: revision routing, metric guard false
positives, revision data loss, the confirm/download gate mismatch, omitted
high-risk findings, business-stage wording, stress category names, data
dictionary feedback and internal keys in user-facing text.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from marvis.agent.service import (
    generate_word_conclusions,
    humanize_internal_terms,
    is_agent_report_revision_intent,
)
from marvis.db import TaskRepository
from marvis.domain import TaskStatus
from marvis.llm_outbound import control_instruction
from marvis.llm_outbound_instructions import user_instruction
from marvis.llm_outbound_revision import (
    ReportRevisionError,
    report_revision_button_field,
    report_revision_request,
)
from marvis.validation_report_copy import (
    infer_scorecard_kind,
    looks_like_metric_rewrite_request,
    scorecard_business_stage,
)
from tests.test_agent_api import (
    REQUIRED_AGENT_CONCLUSIONS,
    _advance_to_writing_artifacts,
    _client,
    _create_task,
)
from tests.test_agent_service import _task
from tests.test_llm_outbound import Wire
from tests.test_llm_outbound_revision import _evidence
from tests.test_llm_outbound_validation import PRIVATE, _native_result


FINAL = "TEXT:final_validation_conclusion"
SUMMARY = "TEXT:pressure_test_summary"
BUTTON = "请根据当前证据修订当前模型报告的「最终验证结论」："
COMPOUND = (
    "把最终验证结论里的“支用申请阶段”改成“贷中额度管理”，"
    "并补充压力测试两类特征剔除后 KS 降到 0 的风险"
)


# ---------- 1. Revision routing ----------


@pytest.mark.parametrize("content", [
    BUTTON,
    BUTTON + "这是行为评分卡，用于贷中额度管理；结论里要写上压力测试两类特征剔除后 KS 降到 0 的风险",
    "请修订最终验证结论",
    "结论里也要写上压力测试的风险",
    "没填的那几项帮我补上，报告里加上压力测试风险",
    "把报告结论补全",
])
def test_revision_intent_covers_button_prefill_and_common_verbs(content):
    assert is_agent_report_revision_intent(content) is True


def test_button_prefix_names_field_and_maps_to_measured_regeneration():
    assert report_revision_button_field(BUTTON + "只评价效果") == FINAL
    request = report_revision_request(user_instruction(BUTTON + "这是行为评分卡，用于贷中额度管理"))
    assert request.operation == "regenerate_from_evidence"
    assert request.target_fields == (FINAL,)
    empty = report_revision_request(user_instruction(BUTTON))
    assert empty.operation == "regenerate_from_evidence" and empty.target_fields == (FINAL,)


def test_button_on_business_field_sets_literal_or_asks_instead_of_inventing():
    scope = "请根据当前证据修订当前模型报告的「适用范围」："
    request = report_revision_request(user_instruction(scope + "贷中额度管理"))
    assert request.operation == "set_local_text" and request.local_text == "贷中额度管理"
    with pytest.raises(ReportRevisionError, match="业务口径字段"):
        report_revision_request(user_instruction(scope + "帮我补充一下"))


def test_replace_plus_append_and_plain_append_are_supported_operations():
    compound = report_revision_request(user_instruction(COMPOUND))
    assert compound.operation == "replace_and_append"
    assert (compound.local_old_text, compound.local_text) == ("支用申请阶段", "贷中额度管理")
    append = report_revision_request(user_instruction("在最终验证结论里补充压力测试的高风险数据源提示"))
    assert append.operation == "append_from_evidence" and append.target_fields == (FINAL,)
    with pytest.raises(ReportRevisionError):
        report_revision_request(user_instruction("在适用范围里补充贷中"))


# ---------- 2. Metric guard ----------


@pytest.mark.parametrize("content, refused", [
    (COMPOUND, False),
    (BUTTON + "结论里要写上剔除后 KS 降到 0 的风险", False),
    ("结论里写上 PSI 超过 0.25 的月份", False),
    ("如何提高KS", False),
    ("把结论里的KS改成0.6", True),
    ("把“KS为1.0”改成“KS为0.9”", True),
    ("AUC从0.7改成0.8", True),
    ("修改一下OOT KS", True),
    ("把KS降到0.3", True),
])
def test_metric_guard_refuses_value_changes_not_mentions(content, refused):
    assert looks_like_metric_rewrite_request(content) is refused


# ---------- 3. Revisions never destroy the report ----------


def _llm_settings(client):
    client.put("/api/settings/llm", json={
        "default_model_id": "m1",
        "models": [{
            "model_id": "m1", "enabled": True, "display_name": "主模型",
            "provider": "OpenAI Compatible", "api_base_url": "https://example.test/v1",
            "model_name": "credit-risk-gpt", "api_key": "secret", "timeout_seconds": 45,
        }],
    })


@pytest.fixture
def drafted(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_job(job_id, settings, task_id, model_profile, opening_message_id, stage=None,
                 stage_message_id=None, acceptance_mode=None, stage_instruction=None):
        calls.append({"stage": stage, "stage_instruction": stage_instruction})

    monkeypatch.setattr("marvis.agent.validation_app_service.run_agent_validation_job", fake_job)
    client = _client(tmp_path)
    _llm_settings(client)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    draft = repo.add_agent_message(
        task_id, role="assistant", stage="word_conclusion_draft", content="草稿",
        metadata={"draft_values": dict(REQUIRED_AGENT_CONCLUSIONS), "report_revision": 0},
    )
    return client, repo, task_id, draft, calls


def test_unparseable_revision_is_clarified_before_any_reset_or_new_draft(drafted):
    client, repo, task_id, draft, calls = drafted
    before = repo.get_report_values(task_id)
    content = "这个是行为评分卡，用在贷中额度管理，不是支用阶段。没填的那几项帮我补上，结论里也要写上压力测试的风险"
    response = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": content, "model_id": "m1"})
    assert response.status_code == 202, response.text
    assert calls == []
    assert repo.get_report_values(task_id) == before
    messages = repo.list_agent_messages(task_id)
    drafts = [message for message in messages if message["stage"] == "word_conclusion_draft"]
    assert [message["id"] for message in drafts] == [draft["id"]]
    assert messages[-1]["content"].startswith("报告修订需要澄清")
    assert messages[-1]["metadata"]["report_fields_unchanged"] is True
    assert repo.get_task(task_id).status == TaskStatus.WRITING_ARTIFACTS


def test_compound_revision_dispatches_current_instruction_and_keeps_baseline(drafted):
    client, repo, task_id, draft, calls = drafted
    response = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": COMPOUND, "model_id": "m1"})
    assert response.status_code == 202, response.text
    assert calls == [{"stage": "word_conclusion_draft", "stage_instruction": COMPOUND}]
    _values, revision = repo.get_report_values(task_id)
    assert revision == 0  # the pending draft stays a valid baseline


def test_generic_rewrite_request_regenerates_instead_of_failing_to_parse(drafted):
    client, repo, task_id, draft, calls = drafted
    content = "报告怎么变成空的了？帮我重新写一下报告草稿"
    response = client.post(f"/api/tasks/{task_id}/agent/messages", json={"content": content, "model_id": "m1"})
    assert response.status_code == 202, response.text
    assert calls == [{"stage": "word_conclusion_draft", "stage_instruction": "重新生成报告"}]


def test_upstream_rerun_drops_conclusions_so_redraft_regenerates_them(tmp_path):
    from marvis.agent.validation_service import reset_agent_task_for_rerun
    from marvis.llm_outbound_revision import revision_fields

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    _advance_to_writing_artifacts(repo, task_id)
    repo.update_agent_report_conclusions(task_id, REQUIRED_AGENT_CONCLUSIONS, expected_revision=0)
    reset_agent_task_for_rerun(repo, task_id, "metrics")
    values, revision = repo.get_report_values(task_id)
    assert revision == 2 and not set(REQUIRED_AGENT_CONCLUSIONS) & set(values)
    fields = revision_fields(report_revision_request(None), values)
    assert set(REQUIRED_AGENT_CONCLUSIONS) <= set(fields)


@pytest.mark.parametrize("version", [1, 2])
def test_replace_and_append_keeps_local_paragraph_and_appends_measured_addendum(monkeypatch, version):
    evidence = _evidence()
    original = f"本模型在支用申请阶段区分良好。{PRIVATE}"
    evidence["saved_report_narrative"]["text_values"][FINAL] = original
    wire = Wire(monkeypatch)
    wire.reply = {FINAL: "压力测试显示剔除后 KS 明显下降，属于!!高风险数据源!!。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=version),
        evidence=evidence, model_profile={}, user_instruction=COMPOUND,
    )
    assert metadata["report_revision_operation"] == "replace_and_append"
    assert values[FINAL].startswith(original.replace("支用申请阶段", "贷中额度管理"))
    assert "高风险数据源" in values[FINAL]
    assert values[SUMMARY] == evidence["saved_report_narrative"]["text_values"][SUMMARY]
    assert len(wire.requests) == 1
    encoded = json.dumps(wire.requests, ensure_ascii=False)
    assert PRIVATE not in encoded and "区分良好" not in encoded


def test_replacement_mismatch_sends_nothing(monkeypatch):
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=2),
        evidence=_evidence(), model_profile={}, user_instruction=COMPOUND,
    )
    assert values == {} and metadata["revision_error_code"] == "revision_text_mismatch"
    assert wire.requests == []


# ---------- 4. Confirm gate matches the download gate ----------


def test_confirm_refuses_empty_required_paragraphs_without_job_or_notice(drafted):
    client, repo, task_id, draft, _calls = drafted
    response = client.post(f"/api/tasks/{task_id}/agent/report-draft/confirm", json={
        "revision": 0, "draft_message_id": draft["id"], "draft_edit_revision": 0,
        "text_values": {**REQUIRED_AGENT_CONCLUSIONS, FINAL: "", SUMMARY: " "},
    })
    assert response.status_code == 422, response.text
    assert "最终验证结论、压力测试总结为空" in response.json()["detail"]
    assert repo.get_active_job_kind(task_id) is None
    assert not any(message["stage"] == "word_conclusion_confirmed" for message in repo.list_agent_messages(task_id))
    assert repo.get_report_values(task_id)[1] == 0


# ---------- 5. Final conclusion names measured high-risk items ----------


def test_final_conclusion_gets_platform_risk_finding_when_model_omits_it(monkeypatch):
    wire = Wire(monkeypatch)
    wire.reply = {FINAL: "模型区分与稳定性良好。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, _metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=1),
        evidence=_evidence(), model_profile={},
        user_instruction=control_instruction("按当前验证结果改写结论，其余不变。"),
    )
    assert values[FINAL].startswith("模型区分与稳定性良好。")
    assert "平台确定性检查提示" in values[FINAL]
    assert "!!压力测试显示" in values[FINAL] and "高风险数据源!!" in values[FINAL]
    assert "!!过拟合检查未通过" in values[FINAL]
    assert "!!OOT 相对 Train 的 PSI 为 0.3600" in values[FINAL]


def test_conclusion_that_already_names_the_risk_is_left_unchanged(monkeypatch):
    from marvis.agent.service import _with_platform_risk_findings

    text = "!!压力测试两类特征为高风险数据源!!；!!过拟合检查未通过!!；OOT PSI 超过阈值，分布迁移明显。"
    values = _with_platform_risk_findings({FINAL: text}, {"validation_results": _native_result()})
    assert values[FINAL] == text


# ---------- 6. Business stage ----------


@pytest.mark.parametrize("name, kind", [
    ("行为评分卡B1", "b"), ("渠道甲B卡", "b"), ("自营T卡", "t"), ("贷前A卡", "a"),
    ("LR模型验证", None), ("A卡与T卡对比", None),
])
def test_scorecard_kind_only_from_explicit_markers(name, kind):
    assert infer_scorecard_kind(name) == kind


def test_latest_user_text_decides_stage_over_model_name():
    assert scorecard_business_stage("贷前A卡", "这是我们新上的行为评分卡") == "b"
    assert scorecard_business_stage("LR模型", "看一下效果") is None


def test_section_prompt_carries_stage_only_when_known():
    from marvis.llm_outbound_validation import validation_report_section_prompt

    known = json.loads(validation_report_section_prompt(_native_result(), topic="overall", business_stage="b"))
    assert known["business_context"] == {"stage": "贷中环节（B卡/行为评分卡）"}
    unknown = json.loads(validation_report_section_prompt(_native_result(), topic="overall"))
    assert "business_context" not in unknown
    assert "不得断言模型所处的授信" in unknown["instructions"]


def test_agent_evidence_records_stage_from_model_name_and_user_turns(tmp_path):
    from types import SimpleNamespace

    from marvis.agent.validation_evidence import task_business_stage

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path)  # model name "A卡"
    repo = TaskRepository(tmp_path / "marvis.sqlite")
    settings = SimpleNamespace(db_path=tmp_path / "marvis.sqlite", tasks_dir=tmp_path / "tasks")
    assert task_business_stage(settings, task_id) == "a"
    repo.add_agent_message(task_id, role="user", stage="chat", content="这是行为评分卡，用在贷中额度管理")
    assert task_business_stage(settings, task_id) == "b"


# ---------- 7. Stress category names ----------


def test_short_category_label_reaches_evidence_but_literals_do_not():
    from marvis.llm_outbound_validation import validation_metrics

    raw = _native_result()
    raw["stress_test"]["per_category"].append({**raw["stress_test"]["per_category"][0], "category": "行为"})
    rows = validation_metrics(raw)["stress_test"]["per_category"]
    assert "category_name" not in rows[0]  # PRIVATE_BUSINESS_LITERAL_739 stays an index
    assert rows[1]["category_name"] == "行为"


# ---------- 8. Data dictionary feedback ----------


def _missing_importance_task(tmp_path):
    from tests.test_material_candidate_recommendations import _create_task as create, _write_materials

    source = _write_materials(tmp_path / "source")
    return create(tmp_path, source=source)


def test_sample_file_is_not_offered_as_dictionary(tmp_path):
    from marvis.api_scan_helpers import material_candidates_payload, validate_material_selection

    task = _missing_importance_task(tmp_path)
    candidates = {item["name"]: item for item in material_candidates_payload(task)["candidates"]["data_dictionary"]}
    assert candidates["sample.csv"]["metadata_compatibility"]["status"] == "incompatible"
    with pytest.raises(ValueError, match="不能选同一个文件"):
        validate_material_selection(task, {
            "notebook_path": "model.ipynb", "sample_path": "sample.csv",
            "pmml_path": "model.pmml", "dictionary_path": "sample.csv",
        })


def test_material_failure_card_reports_missing_importance(tmp_path):
    from marvis.agent.validation_repair import _material_diagnostic

    task = _missing_importance_task(tmp_path)
    diagnostic = _material_diagnostic(task, error="ValueError: selected Notebook path is missing", evidence=None)
    assert diagnostic["code"] == "validation_metadata_incomplete"
    assert diagnostic["title"] == "数据字典不能用"
    assert "缺少 importance（特征重要性）列" in diagnostic["summary"]
    assert "找不到" not in diagnostic["summary"]
    assert any("importance" in action for action in diagnostic["actions"])


def test_dictionary_questions_get_platform_answer_not_invented_columns(tmp_path, monkeypatch):
    from marvis.agent.service import answer_chat_message
    from marvis.api_scan_helpers import material_dictionary_diagnostics

    task = _missing_importance_task(tmp_path)
    evidence = {"material_diagnostics": material_dictionary_diagnostics(task)}

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dictionary requirements must not be generated by a model")

    monkeypatch.setattr("marvis.agent.service._client", forbidden)
    for question in ("数据字典为什么不能用？哪里不完整", "数据字典需要哪些列？"):
        content, metadata = answer_chat_message(
            task=task, user_message=question, conversation=[], evidence=evidence, model_profile={},
        )
        assert metadata["deterministic_answer"] == "dictionary_requirements"
        assert "特征字典.xlsx：不能用，缺少 importance（特征重要性）列" in content
        assert "feature" in content and "category" in content and "importance" in content
        assert "字段类型" not in content and "缺失值定义" not in content


def test_dictionary_reasons_reach_chat_evidence_without_file_names(tmp_path):
    from marvis.api_scan_helpers import material_dictionary_diagnostics
    from marvis.llm_outbound_scan import material_diagnostics_context

    task = _missing_importance_task(tmp_path)
    projected = material_diagnostics_context(material_dictionary_diagnostics(task))
    assert projected["data_dictionary_candidates"] == [
        {"status": "incompatible", "reasons": ["missing_importance_column"], "selected": False},
    ]
    assert "importance" in projected["data_dictionary_requirements"]["importance"]["accepted_column_names"]
    assert "特征字典" not in json.dumps(projected, ensure_ascii=False)


def test_scan_check_message_becomes_metadata_reason_code():
    from marvis.llm_outbound_scan import scan_context

    context = scan_context({"scan": {"checks": [{
        "id": "validation_input_contract", "status": "error",
        "message": "feature metadata: metadata file: missing importance column alias",
    }]}}, workflow_version=2)
    assert context["scan"]["checks"][0]["metadata_reasons"] == ["missing_importance_column"]


# ---------- 9. Internal keys stay out of user text ----------


def test_internal_keys_are_replaced_with_user_labels():
    text = humanize_internal_terms(
        "- Notebook（material_notebook）：通过；`material_data_dictionary` 未满足，"
        "`required_materials_complete = false`；pending_confirmation（needs_confirmation=true，revision 1）；scan_materials"
    )
    assert text == "- Notebook：通过；数据字典 未满足，必需材料尚未齐全；待确认（第 1 版）；材料识别工具"


@pytest.mark.parametrize("workflow_version", [1, 2])
def test_scan_tool_call_message_has_no_tool_name(tmp_path, workflow_version):
    from marvis.agent.validation_app_service import run_agent_scan_stage

    client = _client(tmp_path)
    task_id = _create_task(client, tmp_path, validation_workflow_version=workflow_version)
    repo = TaskRepository(client.app.state.settings.db_path)
    # Empty materials exercise the actual deterministic scan/error path without an LLM.
    if workflow_version == 2:
        with pytest.raises(ValueError, match="selected Notebook path is missing"):
            run_agent_scan_stage(repo, client.app.state.settings, task_id, {})
    else:
        assert run_agent_scan_stage(repo, client.app.state.settings, task_id, {}) is False
    tool_messages = [
        message for message in repo.list_agent_messages(task_id)
        if message.get("metadata", {}).get("tool_call", {}).get("name") == "scan_materials"
    ]
    assert len(tool_messages) == 1
    assert "材料识别工具" in tool_messages[0]["content"]
    assert "scan_materials" not in tool_messages[0]["content"]
    assert tool_messages[0]["metadata"]["tool_call"]["stage"] == "scan"


def test_button_revision_reaches_wire_with_guidance_and_business_stage(monkeypatch):
    evidence = {**_evidence(), "business_stage": "b"}
    wire = Wire(monkeypatch)
    wire.reply = {FINAL: "贷中环节的行为评分卡区分良好，但!!压力测试存在高风险数据源!!。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    instruction = BUTTON + "这是行为评分卡，用于贷中额度管理，不是支用阶段"
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=2),
        evidence=evidence, model_profile={}, user_instruction=instruction,
    )
    assert metadata["report_revision_fields"] == [FINAL] and metadata["confirmable"] is True
    assert len(wire.requests) == 4  # performance, stability, ranking, overall
    for request in wire.requests:
        context = json.loads(request["messages"][1]["content"])
        assert context["business_context"] == {"stage": "贷中环节（B卡/行为评分卡）"}
        assert context["report_revision_operation"]["instruction"] == instruction
    assert values[SUMMARY] == evidence["saved_report_narrative"]["text_values"][SUMMARY]


def test_humanized_scan_failure_still_offers_material_reselection(tmp_path):
    # Eval regression (P0-5): the scan status message is the Chinese guidance,
    # not the English check text, so the dictionary failure card had no
    # 「重新选择验证材料」 option and the user had no way forward.
    from dataclasses import replace
    from types import SimpleNamespace

    from marvis.agent.validation_repair import RESELECT_MATERIALS_COMMAND, diagnose_validation_failure
    from marvis.pipeline_io import SCAN_STAGE_FAILURE_PREFIX
    from marvis.validation_report_copy import MISSING_IMPORTANCE_GUIDANCE

    task = replace(_missing_importance_task(tmp_path), validation_workflow_version=2)
    diagnostic = diagnose_validation_failure(
        settings=SimpleNamespace(db_path=tmp_path / "unused.sqlite"),
        task=task,
        stage="scan",
        error=f"{SCAN_STAGE_FAILURE_PREFIX}{MISSING_IMPORTANCE_GUIDANCE}",
    )
    assert diagnostic is not None
    assert diagnostic["code"] == "validation_metadata_incomplete"
    commands = [option.get("command") for option in diagnostic["repair_options"]]
    assert RESELECT_MATERIALS_COMMAND in commands
    # The card's evidence speaks Chinese: the contract check's platform text
    # ("missing importance column alias") is replaced by the dictionary line.
    scan = {"checks": [{"label": "验证输入契约", "status": "error",
                        "message": "feature metadata: metadata file: missing importance column alias"}]}
    from marvis.agent.validation_repair import _material_diagnostic

    with_scan = _material_diagnostic(task, error="", evidence={"scan": scan})
    values = " ".join(str(item.get("value")) for item in with_scan["evidence"])
    assert "column alias" not in values
    assert "缺少 importance（特征重要性）列" in values


def test_draft_shows_the_algorithm_text_once_the_scan_knows_it():
    # Eval S2 regression: the draft table kept the creation-time placeholder
    # (with the internal RMC_ALGORITHM key) after the scan had found lr.
    from types import SimpleNamespace

    from marvis.agent.validation_stages import _readable_llm_error, _refresh_default_training_description
    from marvis.model_algorithms import model_training_report_text

    key = "TEXT:model_training_description"
    legacy = {key: "待 Notebook 契约 RMC_ALGORITHM 确认后自动生成模型训练说明。", "TEXT:x": "y"}
    refreshed = _refresh_default_training_description(legacy, SimpleNamespace(algorithm="lr"))
    assert refreshed[key] == model_training_report_text("lr") and refreshed["TEXT:x"] == "y"
    own = {key: "我们自己写的训练说明。"}
    assert _refresh_default_training_description(own, SimpleNamespace(algorithm="lr")) == own
    assert _refresh_default_training_description(legacy, SimpleNamespace(algorithm="")) == legacy
    assert _readable_llm_error("LLM request failed") == "大模型请求失败（可能是网络或服务暂时不可用）。"


def test_first_draft_failure_offers_a_retry_and_releases_the_batch_item():
    # Eval S5 regression: an LLM connection failure on the first draft read
    # 「报告修订未完成。LLM request failed」 and left the batch item 执行中.
    from marvis.agent import validation_stages as stages

    class _Repo:
        def __init__(self, drafts):
            self.messages = [{"stage": "word_conclusion_draft"} for _ in range(drafts)]
            self.added = []

        def list_agent_messages(self, _task_id):
            return self.messages

        def add_agent_message(self, task_id, **kwargs):
            self.added.append(kwargs)

    failed = {"report_revision_status": "failed", "llm_error": "LLM request failed"}
    first = _Repo(1)
    stages.add_agent_word_draft_failure_message(first, task_id="t", model_profile={}, metadata=failed)
    message = first.added[-1]
    assert message["content"].startswith("报告结论草稿生成失败。大模型请求失败")
    assert "LLM request failed" not in message["content"]
    assert [reply["command"] for reply in message["metadata"]["quick_replies"]] == ["重新生成结论"]
    later = _Repo(3)
    stages.add_agent_word_draft_failure_message(later, task_id="t", model_profile={}, metadata=failed)
    assert later.added[-1]["content"].startswith("报告修订未完成。")

    from marvis.agent.validation_app_service import is_agent_report_regenerate_intent
    assert is_agent_report_regenerate_intent("重新生成结论")
    source = Path(stages.__file__).read_text(encoding="utf-8")
    failure = source[source.index("add_agent_word_draft_failure_message(\n            repo,"):]
    failure = failure[: failure.index("return False")]
    assert 'status="awaiting_confirmation",\n                stage="word_conclusion_draft",' in failure
