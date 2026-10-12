"""Real protocol bodies for bounded report revisions and local-only baselines."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from marvis.agent.service import generate_word_conclusions
from marvis.llm_outbound import OutboundContextError, control_instruction
from marvis.llm_outbound_revision import ReportRevisionRequest, report_revision_request
from tests.test_agent_service import _task
from tests.test_llm_outbound import Wire
from tests.test_llm_outbound_validation import PRIVATE, _native_result


FINAL = "TEXT:final_validation_conclusion"
SUMMARY = "TEXT:pressure_test_summary"
RECOMMENDATION = "TEXT:pressure_impact_recommendation"


def _evidence():
    return {
        "validation_results": _native_result(),
        "saved_report_narrative": {"report_revision": 4, "text_values": {
            SUMMARY: "原压力摘要 " + PRIVATE, RECOMMENDATION: "原建议 " + PRIVATE,
            FINAL: "原整体结论 " + PRIVATE,
            "TEXT:model_scope": "本地完整业务范围 " + PRIVATE,
            "TEXT:bad_sample_definition": "本地标签定义 " + PRIVATE,
            "TEXT:model_training_description": "本地训练描述 " + PRIVATE,
        }},
        "report_draft": {"message_id": "draft-1", "report_revision": 4, "draft_edit_revision": 2,
                         "text_values": {"TEXT:model_scope": "", "TEXT:good_sample_definition": ""}},
    }


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_safe_targeted_revision_reaches_actual_protocol_body_and_preserves_local_fields(monkeypatch, version, transport):
    evidence = _evidence()
    before = deepcopy(evidence)
    instruction = control_instruction("按当前验证结果改写结论，其余不变。")
    wire = Wire(monkeypatch)
    wire.client.profile["structured_output"] = transport
    wire.reply = {FINAL: "本次 OOT KS 为 0.28，PSI 为 0.36，已测稳定性存在风险。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=version),
        evidence=evidence, model_profile={}, user_instruction=instruction,
    )
    assert metadata["fallback"] is False and metadata["confirmable"] is True
    assert metadata["report_revision_fields"] == [FINAL]
    assert len(wire.requests) == (4 if version == 2 else 1)
    assert evidence == before
    for key, value in evidence["saved_report_narrative"]["text_values"].items():
        if key != FINAL:
            assert values[key] == evidence["report_draft"]["text_values"].get(key, value)
    assert values["TEXT:good_sample_definition"] == ""
    encoded = json.dumps(wire.requests, ensure_ascii=False)
    assert PRIVATE not in encoded and "0.987654321" not in encoded
    for request in wire.requests:
        context = json.loads(request["messages"][1]["content"])
        operation = context["report_revision_operation"]
        assert operation["instruction"] == instruction
        assert operation["target_fields"] == [FINAL]
        assert operation["provenance"] == {"report_revision": 4, "draft_edit_revision": 2}
        assert context["requested_fields"] == [FINAL]
        assert "saved_report_narrative" not in context["evidence"]
        assert "report_draft" not in context["evidence"]
        if version == 1:
            assert "键必须包含" not in request["messages"][0]["content"]
            assert "本次为首次候选文字生成" not in context["instructions"]
    metrics = json.loads(wire.requests[0]["messages"][1]["content"])["evidence"]["validation_results"]
    assert metrics["effectiveness"]["overall"] == before["validation_results"]["effectiveness"]["overall"]


@pytest.mark.parametrize("version", [1, 2])
def test_generic_redraft_preserves_explicitly_empty_conclusion_and_saved_training(monkeypatch, version):
    evidence = _evidence()
    evidence["report_draft"]["text_values"][FINAL] = ""
    wire = Wire(monkeypatch)
    if version == 2:
        wire.replies = [{SUMMARY: "新压力摘要。"}, {RECOMMENDATION: "新建议。"}]
    else:
        wire.reply = {SUMMARY: "新压力摘要。", RECOMMENDATION: "新建议。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=version),
        evidence=evidence, model_profile={},
    )
    assert values[FINAL] == "" and metadata["confirmable"] is False
    assert metadata["fallback"] is False
    assert values["TEXT:model_training_description"] == evidence["saved_report_narrative"]["text_values"]["TEXT:model_training_description"]
    assert len(wire.requests) == (2 if version == 2 else 1)
    assert PRIVATE not in json.dumps(wire.requests)


@pytest.mark.parametrize("instruction", [
    "如果核验通过再改写结论，其余不变。",
    "按当前验证结果改写结论，其余不变。不要继续。",
    "按当前验证结果改写结论，其余不变？",
    "把结论改写成稀疏风险也保留在报告里，其余不变。",
])
def test_approved_control_with_unsupported_semantics_is_clarified_without_losing_constraints(monkeypatch, instruction):
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4), evidence=_evidence(), model_profile={},
        user_instruction=control_instruction(instruction),
    )
    assert values == {} and metadata["outbound_context_status"] == "clarification_required"
    assert metadata["revision_error_code"] == "unsupported_report_instruction"
    assert wire.requests == []


@pytest.mark.parametrize("raw", [PRIVATE, "sk-secret-api-key", "/Users/secret/report.docx", "客户记录:姓名=私密"])
def test_private_user_revision_has_zero_requests_and_explicit_reason(monkeypatch, raw):
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4), evidence=_evidence(), model_profile={}, user_instruction=raw,
    )
    assert values == {} and metadata["confirmable"] is False
    assert metadata["outbound_context_status"] == "clarification_required"
    assert raw not in json.dumps(metadata, ensure_ascii=False) and wire.requests == []


def test_clear_is_local_operation_with_no_model_call(monkeypatch):
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4), evidence=_evidence(), model_profile={},
        user_instruction=control_instruction("删除结论，其余不变。"),
    )
    assert values[FINAL] == "" and values[SUMMARY].startswith("原压力摘要")
    assert metadata["report_revision_operation"] == "clear" and metadata["confirmable"] is False
    assert wire.requests == []


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("instruction, field, replacement", [
    ("把坏样本定义改成 MOB6 逾期 >= 30 天，其他不变。", "TEXT:bad_sample_definition", "MOB6 逾期 >= 30 天"),
    ("请把模型概述改为“用于支用环节风险判断”，其他字段保持不变。", "TEXT:model_overview", "用于支用环节风险判断"),
])
def test_explicit_business_field_replacement_stays_local_and_preserves_every_other_field(monkeypatch, version, instruction, field, replacement):
    evidence = _evidence()
    before = deepcopy(evidence)
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=version),
        evidence=evidence, model_profile={}, user_instruction=instruction,
    )
    assert values[field] == replacement and metadata["report_revision_operation"] == "set_local_text"
    assert metadata["fallback"] is False and metadata["confirmable"] is True
    local = {**evidence["saved_report_narrative"]["text_values"], **evidence["report_draft"]["text_values"]}
    assert all(values[key] == value for key, value in local.items() if key != field)
    assert wire.requests == [] and evidence == before


def test_natural_evidence_rewrite_keeps_complete_user_instruction_at_wire(monkeypatch):
    wire = Wire(monkeypatch)
    wire.reply = {FINAL: "本次 OOT KS 为 0.28，已测稳定性存在风险。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    instruction = "请根据本次验证结果整段重写最终结论，其他段落保持原样。"
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4), evidence=_evidence(), model_profile={},
        user_instruction=instruction,
    )
    assert values and metadata["fallback"] is False
    assert wire.last_context()["report_revision_operation"]["instruction"] == instruction


def test_explicit_business_clear_is_preserved_without_model_request(monkeypatch):
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4), evidence=_evidence(), model_profile={},
        user_instruction="把坏样本定义清空，其余不变。",
    )
    assert values["TEXT:bad_sample_definition"] == ""
    assert values[FINAL].startswith("原整体结论")
    assert metadata["report_revision_operation"] == "clear"
    assert metadata["confirmable"] is True and wire.requests == []


@pytest.mark.parametrize("version", [1, 2])
def test_exact_local_word_replacement_preserves_complete_paragraph_and_metrics(monkeypatch, version):
    evidence = _evidence()
    original = f"适用于授信阶段风险评估。OOT KS 为 0.28，PSI 为 0.36。{PRIVATE}"
    evidence["saved_report_narrative"]["text_values"][FINAL] = original
    before = deepcopy(evidence)
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=version),
        evidence=evidence, model_profile={}, user_instruction='把结论中的“授信阶段”改成“支用阶段”，其余不变。',
    )
    assert values[FINAL] == original.replace("授信阶段", "支用阶段")
    assert metadata["report_revision_operation"] == "replace_local_text"
    assert metadata["confirmable"] is True and metadata["fallback"] is False
    assert evidence == before and wire.requests == []


@pytest.mark.parametrize("instruction, code", [
    ('把结论中的“0.28”改成“0.99”，其余不变。', "metric_rewrite_refused"),
    ('把结论中的“授信阶段”改成“支用阶段”，其余不变。', "revision_text_mismatch"),
])
def test_local_replacement_rejects_metric_changes_and_missing_source_fragment(monkeypatch, instruction, code):
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4), evidence=_evidence(), model_profile={}, user_instruction=instruction,
    )
    assert values == {} and metadata["revision_error_code"] == code
    assert metadata["outbound_context_status"] == "clarification_required" and wire.requests == []


def test_revision_request_cannot_be_minted_from_private_prose_or_arbitrary_constructor():
    with pytest.raises(OutboundContextError):
        report_revision_request(PRIVATE)
    with pytest.raises(OutboundContextError):
        ReportRevisionRequest(None, "regenerate_from_evidence", (FINAL,))


def test_v2_repair_keeps_same_authorized_target_and_does_not_echo_rejected_text(monkeypatch):
    wire = Wire(monkeypatch)
    wire.replies = [PRIVATE, {FINAL: "本次未见可比历史模型，稳定性风险需关注。"}]
    wire.reply = {FINAL: "本次 OOT KS 为 0.28，需结合平台风险判断。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4, validation_workflow_version=2),
        evidence=_evidence(), model_profile={},
        user_instruction=control_instruction("按当前验证结果改写结论，其余不变。"),
    )
    assert values and metadata["fallback"] is False and len(wire.requests) == 5
    first, repair = [json.loads(item["messages"][1]["content"]) for item in wire.requests[:2]]
    assert repair.pop("response_repair")["reason"] == "json_text_contract"
    assert first == repair and PRIVATE not in json.dumps(wire.requests)
