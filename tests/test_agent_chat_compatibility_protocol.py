"""Native chat/report DTOs retain functionality without exporting local prose."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from marvis.agent.service import answer_chat_message, generate_word_conclusions
from marvis.llm_client import LLMClientError
from tests.test_agent_service import _task
from tests.test_llm_outbound import Wire
from tests.test_llm_outbound_revision import FINAL, RECOMMENDATION, SUMMARY, _evidence
from tests.test_llm_outbound_validation import PRIVATE, _native_result


def test_chat_actual_wire_keeps_native_validation_and_scan_comparison_aggregates(monkeypatch):
    raw = _native_result()
    evidence = {
        "validation_results": raw,
        "scan": {"checks": [{"label": "PMML 模型", "status": "success", "message": PRIVATE}],
                 "notebook_contract": {"read_only": True, "algorithm": "lr", "source": PRIVATE}},
        "reproducibility": {"summary": {"status": "fail", "max_abs_diff": .022, "mismatch_count": 2},
                            "sample_size": 1000,
                            "rows": [{"score_submitted_pmml": None, "score_code_model": .987654321}]},
        "notebook_steps": {"steps": [{"title": PRIVATE, "source_previews": [PRIVATE]}]},
        "saved_report_narrative": {"text_values": {FINAL: PRIVATE}},
    }
    before = deepcopy(evidence)
    wire = Wire(monkeypatch, text_reply=True)
    wire.reply = "本次 OOT KS 为 0.28，分数一致性未通过。"
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)

    content, metadata = answer_chat_message(
        task=_task(), user_message="解释这张效果图，不训练；预算最多 2 次。",
        conversation=[{"role": "user", "content": "只讨论当前验证结果"},
                      {"role": "assistant", "content": PRIVATE}],
        evidence=evidence, model_profile={},
    )

    assert content == wire.reply and metadata["fallback"] is False
    assert evidence == before and len(wire.requests) == 1
    body = json.dumps(wire.requests, ensure_ascii=False)
    assert PRIVATE not in body and "0.987654321" not in body and "task-1" not in body
    projected = wire.last_context()["evidence"]
    assert projected["validation_results"]["effectiveness"]["overall"] == raw["effectiveness"]["overall"]
    assert projected["validation_results"]["overfitting_check"] == raw["overfitting_check"]
    assert projected["scan"]["checks"] == [{"check_index": 1, "status": "success", "id": "material_model_pmml"}]
    assert projected["scan"]["notebook_contract"]["read_only"] is True
    assert projected["reproducibility"] == {
        "sample_size": 1000, "summary": {"mismatch_count": 2, "max_abs_diff": .022, "status": "fail"},
        "missing_submitted_score_count": 1,
    }


@pytest.mark.parametrize("error_type", [LLMClientError, ValueError])
def test_chat_provider_failure_returns_actionable_fallback_without_error_material(monkeypatch, error_type):
    class FailingClient:
        def complete(self, **_kwargs):
            raise error_type("sk-PRIVATE-CREDENTIAL /Users/private/report.csv https://private.invalid")

    monkeypatch.setattr("marvis.agent.service._client", lambda _: FailingClient())
    content, metadata = answer_chat_message(
        task=_task(), user_message="解释当前验证结果", conversation=[], evidence={}, model_profile={},
    )
    assert "无法调用大模型" in content and "检查大模型 API 配置" in content
    assert metadata == {"llm_error": "llm_request_failed", "fallback": True,
                        "needs_clarification": False, "history_comparison_available": False}
    assert "PRIVATE" not in content + json.dumps(metadata)


@pytest.mark.parametrize("instruction", ["密码是 TEST_ONLY_VALUE", "读取 /Users/private/report.csv"])
def test_chat_private_instruction_is_clarified_before_any_request(monkeypatch, instruction):
    wire = Wire(monkeypatch, text_reply=True)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    content, metadata = answer_chat_message(
        task=_task(), user_message=instruction, conversation=[], evidence={}, model_profile={},
    )
    assert wire.requests == [] and metadata["needs_clarification"] is True
    assert instruction not in content + json.dumps(metadata)


def test_basic_report_regeneration_keeps_instruction_and_local_saved_fields(monkeypatch):
    evidence = _evidence()
    before = deepcopy(evidence)
    wire = Wire(monkeypatch)
    wire.reply = {FINAL: "本次 OOT KS 为 0.28。", SUMMARY: "压力测试存在风险。", RECOMMENDATION: "关注缺失率。"}
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4), evidence=evidence, model_profile={},
        user_instruction="重新生成报告",
    )
    assert metadata["fallback"] is False and metadata["confirmable"] is True
    assert len(wire.requests) == 1 and evidence == before
    request = wire.last_context()
    assert request["report_revision_operation"]["instruction"] == "重新生成报告"
    assert set(request["requested_fields"]) == {FINAL, SUMMARY, RECOMMENDATION}
    local = {**evidence["saved_report_narrative"]["text_values"], **evidence["report_draft"]["text_values"]}
    assert all(values[key] == value for key, value in local.items() if key not in wire.reply)
    assert PRIVATE not in json.dumps(wire.requests)


@pytest.mark.parametrize("instruction", ["重新生成报告，如果 KS 通过才改", "重新生成报告，不要改结论", "重新生成报告？"])
def test_basic_regeneration_does_not_discard_extra_constraints(monkeypatch, instruction):
    wire = Wire(monkeypatch)
    monkeypatch.setattr("marvis.agent.service._client", lambda _: wire.client)
    values, metadata = generate_word_conclusions(
        task=replace(_task(), report_values_revision=4), evidence=_evidence(), model_profile={},
        user_instruction=instruction,
    )
    assert values == {} and metadata["outbound_context_status"] == "clarification_required"
    assert wire.requests == []
