import json
from dataclasses import replace

from marvis.agent.service import (
    answer_chat_message,
    agent_conclusions_confirmed,
    generate_word_conclusions,
    summarize_stage,
    _strip_agent_response_preamble,
)
from marvis.domain import TaskRecord, TaskStatus
from marvis.llm_client import LLMClientError


def test_strip_agent_response_preamble_removes_only_real_preamble():
    # A genuine instruction-acknowledgement preamble is removed...
    assert _strip_agent_response_preamble(
        "好的，遵照您的指示。以下是针对模型A卡的验证分析。\n\n结论\nPMML 一致。"
    ) == "结论\nPMML 一致。"
    assert _strip_agent_response_preamble(
        "以下是针对模型A卡的阶段分析：\n样本量充足。"
    ) == "样本量充足。"


def test_strip_agent_response_preamble_keeps_content_starting_with_analysis_words():
    # ...but legitimate content whose first clause merely contains 分析/总结
    # mid-sentence must NOT have its opening clause deleted (the keyword must be
    # followed by a real terminator, not 的/，).
    keep1 = "以下是关于本次验证分析的详细内容。结论：通过。"
    keep2 = "以下是基于OOT样本的分析，PSI为0.03，稳定性良好。"
    assert _strip_agent_response_preamble(keep1) == keep1
    assert _strip_agent_response_preamble(keep2) == keep2


def _task() -> TaskRecord:
    return TaskRecord(
        id="task-1",
        model_name="A卡",
        model_version="v1",
        validator="qa",
        source_dir="/tmp/materials",
        algorithm="lgb",
        run_mode="agent",
        target_col="y",
        score_col="pred",
        split_col="split",
        time_col="apply_month",
        feature_columns=[],
        notebook_path=None,
        sample_path=None,
        pmml_path=None,
        dictionary_path=None,
        report_values_revision=0,
        status=TaskStatus.WRITING_ARTIFACTS,
        status_message="metrics generated",
        created_at="2026-05-31T00:00:00",
        updated_at="2026-05-31T00:00:00",
    )


def test_word_conclusion_llm_error_returns_non_confirmable_empty_values(monkeypatch):
    class FailingClient:
        def complete(self, **_kwargs):
            raise LLMClientError("offline")

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: FailingClient(),
    )

    values, metadata = generate_word_conclusions(
        task=_task(),
        evidence={},
        model_profile={"api_base_url": "http://llm", "model_name": "m", "api_key": "k"},
    )

    assert values == {}
    assert agent_conclusions_confirmed(values) is False
    assert metadata["fallback"] is True
    assert metadata["confirmable"] is False
    assert "offline" in metadata["llm_error"]


def test_word_conclusion_uses_non_streaming_json_request(monkeypatch):
    captured = {}

    class CapturingClient:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return json.dumps(
                {
                    "TEXT:pressure_test_summary": "平台压力测试摘要。",
                    "TEXT:pressure_impact_recommendation": "建议关注高影响特征。",
                    "TEXT:final_validation_conclusion": "模型可进入人工复核。",
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: CapturingClient(),
    )

    values, metadata = generate_word_conclusions(
        task=_task(),
        evidence={},
        model_profile={"api_base_url": "http://llm", "model_name": "m", "api_key": "k"},
    )

    assert captured["stream"] is False
    assert captured["response_format"] == {"type": "json_object"}
    assert values["TEXT:final_validation_conclusion"] == "模型可进入人工复核。"
    assert metadata["fallback"] is False


def test_v2_word_conclusion_system_prompt_excludes_legacy_consistency_flow(monkeypatch):
    captured = {}

    class CapturingClient:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return json.dumps(
                {
                    "TEXT:pressure_test_summary": "压力测试摘要。",
                    "TEXT:pressure_impact_recommendation": "压力测试建议。",
                    "TEXT:final_validation_conclusion": "PMML 打分测试完成。",
                },
                ensure_ascii=False,
            )

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: CapturingClient(),
    )
    task = replace(_task(), validation_workflow_version=2)

    generate_word_conclusions(
        task=task,
        evidence={},
        model_profile={"api_base_url": "http://llm", "model_name": "m", "api_key": "k"},
    )

    assert "V2 PMML 打分工作流" in captured["system_prompt"]
    assert "不得使用“可复现”“一致性验证”" in captured["system_prompt"]
    assert "最终验证结论应直接评价模型的区分效果" in captured["system_prompt"]
    assert "不得复述材料扫描" in captured["system_prompt"]
    assert "不得写“可直接部署”或“可直接投产”" in captured["system_prompt"]
    assert "报告已进入" not in captured["system_prompt"]


def test_word_conclusion_invalid_json_reports_format_error(monkeypatch):
    class InvalidJsonClient:
        def complete(self, **_kwargs):
            return "不是 JSON"

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: InvalidJsonClient(),
    )

    values, metadata = generate_word_conclusions(
        task=_task(),
        evidence={},
        model_profile={"api_base_url": "http://llm", "model_name": "m", "api_key": "k"},
    )

    assert values == {}
    assert metadata["fallback"] is True
    assert metadata["confirmable"] is False
    assert metadata["llm_error"] == "LLM 返回不是有效 JSON"


def test_summarize_stage_includes_bounded_memory_context_and_metadata(monkeypatch):
    captured = {}

    class CapturingClient:
        def complete(self, **kwargs):
            captured.update(kwargs)
            return "当前模型 KS 相比历史版本提升，需要继续关注 PSI。"

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: CapturingClient(),
    )
    evidence = {"validation_results": {"effectiveness": {"overall": {"ks": 0.30}}}}
    memory_context = {
        "scope": "cross_task_agent_memory",
        "memories": [
            {
                "id": "mem-1",
                "memory_type": "model_experience",
                "summary": "上一版A卡模型在202601自营渠道KS为20。",
                "payload": {
                    "ks": 20,
                    "auc": 0.68,
                    "psi": 0.06,
                    "month": "202601",
                    "channel": "自营",
                    "model_name": "A卡模型",
                    "model_version": "V2025",
                },
                "source_task_id": "task-old",
                "confidence": "high",
                "match_reason": "exact model",
            }
        ],
    }

    content, metadata = summarize_stage(
        task=_task(),
        stage="metrics",
        evidence=evidence,
        memory_context=memory_context,
        model_profile={"model_id": "m1"},
        fallback="fallback",
    )

    prompt = json.loads(captured["user_prompt"])
    assert captured["max_tokens"] == 4096
    assert content == "当前模型 KS 相比历史版本提升，需要继续关注 PSI。"
    assert "压力测试风险必须完整覆盖" in prompt["instructions"]
    assert "不得停在半句话" in prompt["instructions"]
    assert prompt["cross_task_memory"]["memories"][0]["id"] == "mem-1"
    assert "不能改变" in prompt["cross_task_memory"]["usage_rules"]
    assert metadata["memory_references"] == [
        {
            "kind": "raw",
            "id": "mem-1",
            "memory_type": "model_experience",
            "source_task_id": "task-old",
            "confidence": "high",
            "use_reason": "metrics",
        }
    ]
    assert evidence == {"validation_results": {"effectiveness": {"overall": {"ks": 0.30}}}}


def test_summarize_stage_fallback_does_not_claim_memory_use(monkeypatch):
    class FailingClient:
        def complete(self, **_kwargs):
            raise LLMClientError("offline")

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: FailingClient(),
    )

    _, metadata = summarize_stage(
        task=_task(),
        stage="metrics",
        evidence={},
        memory_context={"memories": [{"id": "mem-1", "memory_type": "model_experience"}]},
        model_profile={"model_id": "m1"},
        fallback="fallback",
    )

    assert metadata["fallback"] is True
    assert "memory_references" not in metadata


def test_summarize_stage_truncates_memory_summary_in_prompt(monkeypatch):
    captured = {}

    class CapturingClient:
        def complete(self, **kwargs):
            captured["user_prompt"] = kwargs["user_prompt"]
            return "已参考历史偏好。"

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: CapturingClient(),
    )

    summarize_stage(
        task=_task(),
        stage="metrics",
        evidence={},
        memory_context={
            "memories": [
                {
                    "id": "mem-long",
                    "memory_type": "user_preference",
                    "summary": "报告措辞保持克制。" * 80,
                    "source_task_id": "task-old",
                }
            ],
        },
        model_profile={"model_id": "m1"},
        fallback="fallback",
    )

    prompt = json.loads(captured["user_prompt"])
    summary = prompt["cross_task_memory"]["memories"][0]["summary"]
    assert len(summary) <= 403
    assert summary.endswith("...")


def test_answer_chat_message_memory_context_is_separate_from_task_conversation(monkeypatch):
    captured = {}

    class CapturingClient:
        def complete(self, **kwargs):
            captured["user_prompt"] = kwargs["user_prompt"]
            return "上一版记录显示 KS 更低，当前版本效果更好。"

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: CapturingClient(),
    )

    _, metadata = answer_chat_message(
        task=_task(),
        user_message="和之前模型比怎么样？",
        conversation=[],
        evidence={"validation_results": {"effectiveness": {"overall": {"ks": 30}}}},
        memory_context={
            "memories": [
                {
                    "id": "mem-2",
                    "memory_type": "model_experience",
                    "summary": "历史模型KS为20。",
                    "source_task_id": "task-history",
                    "confidence": "medium",
                }
            ],
        },
        model_profile={"model_id": "m1"},
    )

    prompt = json.loads(captured["user_prompt"])
    assert prompt["cross_task_memory"]["memories"][0]["id"] == "mem-2"
    assert prompt["conversation_memory"]["scope"] == "same_agent_task"
    assert metadata["memory_references"][0]["use_reason"] == "chat"


def test_answer_chat_message_fallback_does_not_claim_memory_use(monkeypatch):
    class EmptyClient:
        def complete(self, **_kwargs):
            return ""

    monkeypatch.setattr(
        "marvis.agent.service._client",
        lambda _profile: EmptyClient(),
    )

    _, metadata = answer_chat_message(
        task=_task(),
        user_message="和之前模型比怎么样？",
        conversation=[],
        evidence={},
        memory_context={"memories": [{"id": "mem-2", "memory_type": "model_experience"}]},
        model_profile={"model_id": "m1"},
    )

    assert metadata["fallback"] is True
    assert "memory_references" not in metadata
