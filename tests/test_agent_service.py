import json

from riskmodel_checker.agent.service import (
    answer_chat_message,
    agent_conclusions_confirmed,
    generate_word_conclusions,
    summarize_stage,
)
from riskmodel_checker.domain import TaskRecord, TaskStatus
from riskmodel_checker.llm_client import LLMClientError


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
        "riskmodel_checker.agent.service._client",
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


def test_summarize_stage_includes_bounded_memory_context_and_metadata(monkeypatch):
    captured = {}

    class CapturingClient:
        def complete(self, **kwargs):
            captured["user_prompt"] = kwargs["user_prompt"]
            return "当前模型 KS 相比历史版本提升，需要继续关注 PSI。"

    monkeypatch.setattr(
        "riskmodel_checker.agent.service._client",
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
    assert content == "当前模型 KS 相比历史版本提升，需要继续关注 PSI。"
    assert prompt["cross_task_memory"]["memories"][0]["id"] == "mem-1"
    assert "不能改变" in prompt["cross_task_memory"]["usage_rules"]
    assert metadata["memory_references"] == [
        {
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
        "riskmodel_checker.agent.service._client",
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


def test_answer_chat_message_memory_context_is_separate_from_task_conversation(monkeypatch):
    captured = {}

    class CapturingClient:
        def complete(self, **kwargs):
            captured["user_prompt"] = kwargs["user_prompt"]
            return "上一版记录显示 KS 更低，当前版本效果更好。"

    monkeypatch.setattr(
        "riskmodel_checker.agent.service._client",
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
        "riskmodel_checker.agent.service._client",
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
