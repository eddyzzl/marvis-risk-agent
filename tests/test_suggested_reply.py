"""UI-6: one display-only suggested next reply per LLM-authored Agent turn."""

import json
from pathlib import Path

import pytest

from marvis.agent.service import answer_chat_message
from marvis.agent.suggested_reply import (
    MAX_SUGGESTED_REPLY_CHARS,
    attach_suggested_reply,
    sanitize_suggested_reply,
    split_suggested_reply,
)
from marvis.agent.workflow_insights import render_workflow_insight
from tests.test_agent_service import _task


@pytest.mark.parametrize("value", [
    "确认", "确认，继续下一步", "把调参轮数改为 50", "为什么 OOT 表现下降？", "Continue with the next step",
])
def test_plain_short_suggestions_survive_verbatim(value):
    assert sanitize_suggested_reply(value) == value


def test_wrapping_quotes_and_whitespace_are_trimmed():
    assert sanitize_suggested_reply("  「 确认 」 ") == "确认"
    assert sanitize_suggested_reply("确认\n第二行被丢弃") == "确认"


@pytest.mark.parametrize("value", [None, "", "   ", 42, ["确认"], "好" * (MAX_SUGGESTED_REPLY_CHARS + 1)])
def test_unusable_values_are_dropped(value):
    assert sanitize_suggested_reply(value) is None


# Rule 1: never metric values or result numbers.
@pytest.mark.parametrize("value", [
    "KS 0.42 已达标，继续", "AUC=0.71 可以吗", "PSI 为 0.3 需要处理吗", "坏账率 3% 偏高吗",
    "OOT 达到 0.35 了", "命中率仅 62", "保留 18 分以上的客户", "百分之五的样本",
])
def test_metric_values_and_result_numbers_are_dropped(value):
    assert sanitize_suggested_reply(value) is None


# Rule 3: no internal codes unless the user typed them.
@pytest.mark.parametrize("value", [
    "用 ds_8f2a 重新训练", "继续 plan_42", "打开 artifact-77", "调用 modeling.train_model",
    "把 oot_ks 设成目标", "看一下 3f9c2a1b7d4e", "引用 $ref:step.output.x", "`确认`", "打开 /Users/x/a",
])
def test_internal_codes_are_dropped(value):
    assert sanitize_suggested_reply(value) is None


def test_field_codes_the_user_typed_are_allowed():
    assert sanitize_suggested_reply("把 sample_weight 换成 w2", user_message="sample_weight 用什么？") \
        == "把 sample_weight 换成 w2"
    assert sanitize_suggested_reply("把 sample_weight 换成 w2", user_message="换个权重") is None


def test_attach_only_sets_a_usable_value():
    assert attach_suggested_reply({}, "确认") == {"suggested_reply": "确认"}
    assert attach_suggested_reply({"suggested_reply": "旧"}, "KS 0.4") == {}
    assert attach_suggested_reply({}, None) == {}


def test_split_only_takes_the_trailing_line():
    body, raw = split_suggested_reply("结论如下。\n\n建议回复：确认")
    assert (body, raw) == ("结论如下。", "确认")
    body, raw = split_suggested_reply("正文提到「建议回复：确认」这个词\n最后一行正文")
    assert raw is None and body.endswith("最后一行正文")
    assert split_suggested_reply("**建议回复**：为什么 PSI 偏高？")[1] == "为什么 PSI 偏高？"


class _Chat:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def complete(self, **kwargs):
        self.prompts.append(json.loads(kwargs["user_prompt"]))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _chat(monkeypatch, reply, message="这个模型稳定吗？"):
    client = _Chat(reply)
    monkeypatch.setattr("marvis.agent.service._client", lambda _profile: client)
    deltas = []
    content, metadata = answer_chat_message(
        task=_task(), user_message=message, conversation=[], evidence={},
        model_profile={"model_id": "m1"}, on_delta=deltas.append,
    )
    return client, content, metadata, deltas


def test_chat_turn_returns_suggestion_as_metadata_not_prose(monkeypatch):
    client, content, metadata, deltas = _chat(
        monkeypatch, "稳定性可接受，OOT 与训练集分布接近。\n建议回复：压力测试结果怎么样？")
    assert metadata["suggested_reply"] == "压力测试结果怎么样？"
    assert "建议回复" not in content and "建议回复" not in "".join(deltas)
    assert content == "稳定性可接受，OOT 与训练集分布接近。"
    # Same single request carries the contract; no second call is made.
    assert len(client.prompts) == 1
    contract = client.prompts[0]["suggested_reply_contract"]
    assert "相同的语言" in contract and "等待用户确认" in contract and "指标数值" in contract


@pytest.mark.parametrize("line", ["建议回复：KS 0.42 够用吗", "建议回复：继续 plan_42", "建议回复：" + "好" * 41])
def test_chat_drops_unsafe_suggestion_but_keeps_the_turn(monkeypatch, line):
    _, content, metadata, _ = _chat(monkeypatch, "当前结果已生成。\n" + line)
    assert "suggested_reply" not in metadata
    assert content == "当前结果已生成。"
    assert metadata["fallback"] is False


def test_chat_without_suggestion_or_with_failure_omits_the_key(monkeypatch):
    from marvis.llm_client import LLMClientError

    _, content, metadata, _ = _chat(monkeypatch, "当前结果已生成。")
    assert content == "当前结果已生成。" and "suggested_reply" not in metadata
    _, _, metadata, _ = _chat(monkeypatch, LLMClientError("offline"))
    assert metadata["fallback"] is True and "suggested_reply" not in metadata
    _, _, metadata, _ = _chat(monkeypatch, "建议回复：确认")
    assert metadata.get("empty_llm_response") is True and "suggested_reply" not in metadata


def _insight_context(**extra):
    return {
        "title": "特征分析解读", "milestone": "feature_analyzed", "facts": ["已分析 3 个特征"],
        "risks": [], "recommendations": [], "evidence": [],
        "measured_evidence": {"feature_count": 3, "features": [{"feature": "x1", "state": "recommended"}]},
        **extra,
    }


class _Insight:
    profile = {"model_name": "fake"}

    def __init__(self, suggestion):
        self.suggestion = suggestion
        self.prompts = []

    def complete(self, **kwargs):
        self.prompts.append(json.loads(kwargs["user_prompt"]))
        return json.dumps({"summary": "整体可用。", "findings": [], "risks": [], "recommendations": [],
                           "suggested_reply": self.suggestion}, ensure_ascii=False)


def test_gate_insight_asks_for_a_confirmation_style_suggestion():
    client = _Insight("确认")
    result = render_workflow_insight(_insight_context(awaiting_confirmation=True), client=client)
    assert result["suggested_reply"] == "确认"
    prompt = client.prompts[0]
    assert prompt["awaiting_confirmation"] is True
    assert "suggested_reply" in prompt["instruction"]
    other = _Insight("下一步做什么？")
    render_workflow_insight(_insight_context(), client=other)
    assert other.prompts[0]["awaiting_confirmation"] is False


def test_templated_insight_fallback_has_no_suggestion():
    result = render_workflow_insight(_insight_context(awaiting_confirmation=True), client=None)
    assert result["generated_by"] == "deterministic_fallback"
    assert "suggested_reply" not in result


# Rule 2: display-only. Only the producing adapters mention the key; nothing
# on the backend reads it to decide, route, confirm or execute anything.
def test_suggested_reply_is_never_read_by_backend_logic():
    root = Path(__file__).resolve().parents[1] / "marvis"
    producers = {
        "agent/suggested_reply.py", "agent/service.py", "agent/workflow_insights.py",
        "agent/turn_handlers/shared.py", "llm_outbound_workflows.py", "llm_outbound_chat.py",
    }
    users = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "suggested_reply" in path.read_text(encoding="utf-8")
    }
    assert users <= producers, users - producers


# Codes glued to Chinese text: ``\b`` treats CJK as word characters, so the
# boundaries must be ASCII-only.
@pytest.mark.parametrize("value", [
    "删除任务task_3f2a9c1d吗", "看下ds_ab12数据集", "用版本3f2a9c1d8e重跑", "调用modeling.train_model吧",
    "把oot_ks设成目标", "引用abcdef12的结果",
])
def test_codes_glued_to_chinese_text_are_dropped(value):
    assert sanitize_suggested_reply(value) is None


@pytest.mark.parametrize("value", ["模型KS达到零点四二吗", "AUC 只有零点六", "PSI为一点二需要处理吗", "通过率六十左右可以吗"])
def test_metric_values_in_chinese_numerals_are_dropped(value):
    assert sanitize_suggested_reply(value) is None


def test_chinese_words_without_codes_or_values_still_pass():
    for value in ("看一下分箱结果", "把调参轮数改为五十", "为什么模型表现下降？"):
        assert sanitize_suggested_reply(value) == value


@pytest.mark.parametrize("value", ["确认", "确认，继续下一步", "好的，同意", "Approve", "continue"])
def test_required_human_decision_gate_never_gets_an_approval_suggestion(value):
    assert sanitize_suggested_reply(value, human_decision_required=True) is None
    assert attach_suggested_reply({"human_decision_gate": "required"}, value) == {"human_decision_gate": "required"}
    assert attach_suggested_reply({"human_decision_gate": "none"}, value)["suggested_reply"] == value


def test_required_gate_still_gets_a_question_or_adjustment():
    for value in ("为什么要剔除这些特征？", "把阈值调严一些"):
        assert sanitize_suggested_reply(value, human_decision_required=True) == value
