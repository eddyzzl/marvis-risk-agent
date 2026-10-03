"""Bounded report sections assemble atomically without changing measured evidence."""
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from marvis.agent.service import generate_word_conclusions
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.agent.validation_narrative import generate_v2_sections
from test_agent_service import _task


VALUES = {
    "TEXT:pressure_test_summary": "剔除外部特征后 KS 从 0.31 降至 0.19，存在高风险依赖。",
    "TEXT:pressure_impact_recommendation": "对高风险数据源设置可用性监控并复核替代特征。",
    "TEXT:final_validation_conclusion": "OOT KS 为 0.31，本次未见可比历史模型；需关注数据源依赖。",
    "TEXT:model_training_description": "本模型采用逻辑回归，材料未提供正则化参数。",
}


def run(monkeypatch, mutate=None, saved=None):
    calls = []
    evidence = {"validation_results": {"effectiveness": {"overall": {"ks": 0.31}}}}
    if saved is not None:
        evidence["report_draft"] = {"text_values": saved}
    original = deepcopy(evidence)

    class Client:
        def complete(self, **kwargs):
            calls.append(kwargs)
            schema = kwargs.get("json_schema", {}).get("schema", {})
            keys = schema.get("required", list(VALUES))
            value = {key: VALUES[key] for key in keys}
            if mutate:
                value = mutate(len(calls), value)
            return json.dumps(value, ensure_ascii=False)

    monkeypatch.setattr("marvis.agent.service._client", lambda _: Client())
    result = generate_word_conclusions(
        task=replace(_task(), validation_workflow_version=2),
        evidence=evidence, model_profile={}, user_instruction="保留现有业务事实，明确数据源风险。",
    )
    assert evidence == original
    return calls, result


def test_report_sections_each_use_their_own_typed_contract_and_same_task(monkeypatch):
    calls, (values, metadata) = run(monkeypatch)
    assert len(calls) == 4
    assert {key: values[key] for key in VALUES} == VALUES
    assert metadata["fallback"] is False
    for call in calls:
        assert call["stream"] is False
        assert call["max_tokens"] == 2048
        assert call["caller"] == "validation_report_section"
        assert call["prompt_name"] == "WORD_CONCLUSION_V2_SYSTEM_PROMPT"
        payload = json.loads(call["user_prompt"])
        assert payload["task"]["model_name"] == "A卡"
        assert payload["user_instruction"] == "保留现有业务事实，明确数据源风险。"
        schema = call["json_schema"]["schema"]
        assert len(schema["required"]) == 1
        assert schema["additionalProperties"] is False


@pytest.mark.parametrize("bad", [None, True, 1, [], {"invented": "text"}, "   "])
def test_malformed_section_is_never_coerced_to_report_text(monkeypatch, bad):
    _, (values, metadata) = run(
        monkeypatch, lambda _n, value: {key: bad for key in value},
    )
    assert values == {}
    assert metadata["fallback"] is True and metadata["confirmable"] is False


def test_later_section_failure_never_exposes_a_partially_confirmable_draft(monkeypatch):
    def fail_last(n, value):
        if n == 4:
            raise LLMClientError("synthetic last section unavailable")
        return value
    calls, (values, metadata) = run(monkeypatch, fail_last)
    assert len(calls) == 4
    assert values == {} and metadata["confirmable"] is False


def test_cross_section_revision_keeps_complete_current_draft_and_explicit_empty_fields():
    draft = {"text_values": {"TEXT:model_scope": "仅限既有支用客群", "TEXT:sample_audience": ""}}
    original = {"stage": "word_conclusion_draft", "evidence": {"report_draft": draft},
                "user_instruction": "把当前适用范围同步到最终结论，保持样本客群字段为空。"}
    seen = []

    class Client:
        def complete(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            seen.append(payload)
            assert payload["evidence"]["report_draft"] == draft
            keys = kwargs["json_schema"]["schema"]["properties"]
            return json.dumps({key: VALUES[key] for key in keys if key in VALUES})

    generate_v2_sections(Client(), json.dumps(original))
    assert len(seen) == 4
    assert original["evidence"]["report_draft"] == draft


def test_section_omission_preserves_saved_narrative_and_explicit_blanks(monkeypatch):
    saved = {"TEXT:model_scope": "仅限原有支用客户", "TEXT:bad_sample_definition": "MOB3 DPD60",
             "TEXT:sample_audience": ""}
    _, (values, metadata) = run(monkeypatch, saved=saved)
    assert metadata["fallback"] is False
    assert {key: values[key] for key in saved} == saved


def test_explicit_optional_section_edit_overrides_saved_value_including_clear(monkeypatch):
    def revise(n, value):
        if n == 4:
            return {**value, "TEXT:model_scope": "", "TEXT:sample_audience": "用户指定的新客群"}
        return value
    _, (values, metadata) = run(monkeypatch, revise, saved={
        "TEXT:model_scope": "旧范围", "TEXT:sample_audience": "",
    })
    assert metadata["fallback"] is False
    assert values["TEXT:model_scope"] == ""
    assert values["TEXT:sample_audience"] == "用户指定的新客群"


@pytest.mark.parametrize("transport", ["json_schema", "json_object", "none"])
def test_real_client_adapts_each_section_to_provider_transport(monkeypatch, transport):
    requests = []

    class Reply:
        def __init__(self, value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def __iter__(self):
            yield json.dumps({"choices": [{"message": {"content": json.dumps(self.value)}}]}).encode()

    def provider(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        requested = json.loads(body["messages"][1]["content"])["requested_fields"]
        if transport == "json_schema":
            schema = body["response_format"]["json_schema"]["schema"]
        else:
            schema = json.loads(body["messages"][0]["content"].split("\n")[-1])["schema"]
            if transport == "none":
                assert "response_format" not in body
            else:
                assert body["response_format"] == {"type": "json_object"}
        assert set(schema["properties"]) == set(requested)
        assert schema["required"] == [requested[0]]
        assert schema["additionalProperties"] is False
        return Reply({requested[0]: VALUES[requested[0]]})

    monkeypatch.setattr("marvis.llm_client.urlopen", provider)
    client = OpenAICompatibleLLMClient({
        "api_base_url": "https://example.test/v1", "api_key": "synthetic-key",
        "model_name": "synthetic-model", "structured_output": transport,
    })
    values = generate_v2_sections(client, json.dumps({"evidence": {}}))
    assert values == VALUES
    assert len(requests) == 4
