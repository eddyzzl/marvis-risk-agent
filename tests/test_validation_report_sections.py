"""Bounded report sections assemble atomically without changing measured evidence."""
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from marvis.agent.service import generate_word_conclusions
from marvis.llm_client import LLMClientError, OpenAICompatibleLLMClient
from marvis.agent.validation_narrative import generate_v2_metrics_summary, generate_v2_sections
from test_agent_service import _task


VALUES = {
    "TEXT:pressure_test_summary": "剔除外部特征后 KS 从 0.31 降至 0.19，存在高风险依赖。",
    "TEXT:pressure_impact_recommendation": "对高风险数据源设置可用性监控并复核替代特征。",
    "TEXT:final_validation_conclusion": "OOT KS 为 0.31，本次未见可比历史模型；需关注数据源依赖。",
    "TEXT:model_training_description": "本模型采用逻辑回归，材料未提供正则化参数。",
}
TOPIC_TEXT = {
    "performance": "Train/Test/OOT KS 为 0.34/0.32/0.31，过拟合需按平台检查判断。",
    "stability": "OOT PSI 为 0.06，当前未见明显分布迁移。",
    "ranking": "按 train 分箱与独立分箱均需检查单调性、头尾幅度及区分度。",
    "overall": VALUES["TEXT:final_validation_conclusion"],
}
ASSEMBLED = {**VALUES, "TEXT:final_validation_conclusion": "\n\n".join(TOPIC_TEXT.values())}


def reply_for(request):
    keys = request["json_schema"]["schema"]["required"]
    topic = json.loads(request["user_prompt"])["narrative_topic"]
    return {key: TOPIC_TEXT[topic] if key == "TEXT:final_validation_conclusion" else VALUES[key] for key in keys}


def run(monkeypatch, mutate=None, saved=None):
    calls = []
    evidence = {"validation_results": {"effectiveness": {"overall": {"ks": 0.31}}}}
    if saved is not None:
        evidence["report_draft"] = {"text_values": saved}
    original = deepcopy(evidence)

    class Client:
        def complete(self, **kwargs):
            calls.append(kwargs)
            value = reply_for(kwargs)
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
    assert len(calls) == 7
    assert {key: values[key] for key in VALUES} == ASSEMBLED
    assert metadata["fallback"] is False
    for call in calls:
        assert call["stream"] is False
        assert call["max_tokens"] == 2048
        assert call["caller"] == "validation_report_" + json.loads(call["user_prompt"])["narrative_topic"]
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
        if n == 7:
            raise LLMClientError("synthetic last section unavailable")
        return value
    calls, (values, metadata) = run(monkeypatch, fail_last)
    assert len(calls) == 7
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
            return json.dumps(reply_for(kwargs))

    generate_v2_sections(Client(), json.dumps(original))
    assert len(seen) == 7
    assert original["evidence"]["report_draft"] == draft


def test_section_omission_preserves_saved_narrative_and_explicit_blanks(monkeypatch):
    saved = {"TEXT:model_scope": "仅限原有支用客户", "TEXT:bad_sample_definition": "MOB3 DPD60",
             "TEXT:sample_audience": ""}
    _, (values, metadata) = run(monkeypatch, saved=saved)
    assert metadata["fallback"] is False
    assert {key: values[key] for key in saved} == saved


def test_explicit_optional_section_edit_overrides_saved_value_including_clear(monkeypatch):
    def revise(n, value):
        if n == 7:
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
        topic = json.loads(body["messages"][1]["content"])["narrative_topic"]
        return Reply({requested[0]: TOPIC_TEXT[topic] if requested[0] == "TEXT:final_validation_conclusion" else VALUES[requested[0]]})

    monkeypatch.setattr("marvis.llm_client.urlopen", provider)
    client = OpenAICompatibleLLMClient({
        "api_base_url": "https://example.test/v1", "api_key": "synthetic-key",
        "model_name": "synthetic-model", "structured_output": transport,
    })
    values = generate_v2_sections(client, json.dumps({"evidence": {}}))
    assert values == ASSEMBLED
    assert len(requests) == 7


def test_topics_keep_exact_evidence_and_memory_without_repeating_raw_bin_tables():
    metrics = {
        "algorithm": "logistic", "basic_info": {"hyperparameters": {"C": 0.25}},
        "effectiveness": {"overall": [{"split": "oot", "ks": 0.31, "auc": 0.71}],
                          "monthly_psi": [{"month": "2026-01", "psi": 0.36}],
                          "lift_ranking_assessment": {"splits": {"oot": {"status": "weak"}}},
                          "bin_tables": {"oot": [{"private_bulk": "x" * 10000}]}},
        "stress_test": {"baseline": {"ks": 0.31}, "per_category": [{"ks": 0.19}]},
        "overfitting_check": {"status": "pass"},
    }
    original = {"stage": "word_conclusion_draft", "evidence": {"validation_results": metrics},
                "cross_task_memory": {"references": [{"id": "historical-1", "source_task_id": "old"}]}}
    before = deepcopy(original)
    seen = {}

    class Client:
        def complete(self, **kwargs):
            payload = json.loads(kwargs["user_prompt"])
            topic = payload["narrative_topic"]
            assert payload["cross_task_memory"] == original["cross_task_memory"]
            seen[topic] = payload["evidence"]["validation_results"]
            assert "private_bulk" not in kwargs["user_prompt"]
            return json.dumps(reply_for(kwargs))

    generate_v2_sections(Client(), json.dumps(original))
    assert original == before
    for topic in ("performance", "stability", "ranking", "overall"):
        assert seen[topic]["effectiveness"]["overall"] == metrics["effectiveness"]["overall"]
    for topic in ("stability", "overall", "recommendation"):
        assert seen[topic]["effectiveness"]["monthly_psi"] == metrics["effectiveness"]["monthly_psi"]
    assert "monthly_psi" not in seen["performance"]["effectiveness"]
    assert seen["ranking"]["effectiveness"]["lift_ranking_assessment"] == metrics["effectiveness"]["lift_ranking_assessment"]
    assert "stress_test" not in seen["ranking"]
    assert seen["overall"]["stress_test"] == metrics["stress_test"]
    assert seen["training"]["basic_info"]["hyperparameters"] == {"C": 0.25}


def test_v2_metrics_uses_complete_topics_and_only_streams_a_validated_whole(monkeypatch):
    from marvis.agent.service import summarize_stage

    calls, deltas = [], []

    class Client:
        def complete(self, **kwargs):
            calls.append(kwargs)
            return json.dumps({"summary": json.loads(kwargs["user_prompt"])["narrative_topic"] + " 的完整证据解释。"})

    monkeypatch.setattr("marvis.agent.service._client", lambda _: Client())
    content, metadata = summarize_stage(
        task=replace(_task(), validation_workflow_version=2), stage="metrics",
        evidence={}, model_profile={}, fallback="指标已计算。", on_delta=deltas.append,
    )
    assert metadata["fallback"] is False
    assert deltas == [content] and len(calls) == 6
    for heading in ("总体判断", "效果表现", "稳定性表现", "压力测试风险", "建议"):
        assert heading + "\n" in content
    assert "performance 的完整证据解释。\n\nranking 的完整证据解释。" in content
    assert all(call["max_tokens"] == 2048 and call["caller"].startswith("validation_metrics_") for call in calls)


def test_v2_metrics_incomplete_json_is_not_exposed_as_a_partial_summary(monkeypatch):
    from marvis.agent.service import summarize_stage

    deltas = []

    class Client:
        def complete(self, **kwargs):
            topic = json.loads(kwargs["user_prompt"])["narrative_topic"]
            return '{"summary":' if topic == "recommendation" else json.dumps({"summary": topic})

    monkeypatch.setattr("marvis.agent.service._client", lambda _: Client())
    content, metadata = summarize_stage(
        task=replace(_task(), validation_workflow_version=2), stage="metrics",
        evidence={}, model_profile={}, fallback="指标已计算。", on_delta=deltas.append,
    )
    assert metadata["fallback"] is True and "recommendation" in metadata["llm_error"]
    assert deltas == [] and "overall" not in content
    with pytest.raises(ValueError, match="recommendation"):
        generate_v2_metrics_summary(Client(), json.dumps({"stage": "metrics", "evidence": {}}))
