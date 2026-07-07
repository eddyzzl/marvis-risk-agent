"""Unit tests for agent intent matchers.

Pure helper tests — no FastAPI app, no LLM, no DB. They guard the
phrasings that route a user reply into the "advance to next stage"
branch instead of the "treat as chat question" branch.
"""

import pytest

from marvis.agent.service import (
    is_agent_advance_intent,
    is_continue_validation_intent,
    is_start_validation_intent,
)


@pytest.mark.parametrize(
    "phrasing",
    [
        # Bare phrasings already supported — regression guards.
        "继续",
        "继续吧",
        "请继续",
        "下一步",
        "继续验证",
        "继续执行",
        # Affix-stripped phrasings already supported.
        "好的继续",
        "先继续吧",
        "请继续吧",
        # The bug report: user types an acknowledgment before "继续验证吧",
        # separated by a Chinese comma. The matcher must still route it as
        # advance-intent, otherwise the agent acknowledges but never starts
        # the next stage.
        "明白了，继续验证吧",
        # Adjacent acknowledged-continue phrasings — same fix should cover.
        "明白了继续验证吧",
        "好的，继续验证",
        "嗯，继续验证",
        "知道了，继续",
        "了解，请继续",
        "收到，继续验证",
        "懂了，继续执行下一步",
    ],
)
def test_continue_intent_recognized(phrasing):
    assert is_continue_validation_intent(phrasing), phrasing
    assert is_agent_advance_intent(phrasing), phrasing


@pytest.mark.parametrize(
    "phrasing",
    [
        # Explicit negations must keep returning False.
        "不继续",
        "先不继续",
        "暂不继续",
        "不要继续",
        "不要继续验证",
        "别继续",
        "无需继续",
        # Broader negation family: the substring fallback would otherwise
        # return True on these because "继续验证" / "继续执行" appears inside.
        "不需要继续验证吧",
        "不想继续验证",
        "没必要继续验证",
        "不打算继续验证",
        "我不会继续执行下一步",
        "暂时不继续验证",
        # Genuinely chat-only phrasings — questions, comments, no advance.
        "为什么 PMML 比 pickle 好？",
        "Notebook 的 step-1 是什么意思",
        "再帮我解释一下指标计算",
        "",
        "   ",
    ],
)
def test_non_continue_inputs_stay_chat(phrasing):
    assert not is_continue_validation_intent(phrasing), phrasing
    # The advance gate (validation_agent.py:219) is what actually dispatches a
    # run, so a chat-only 继续-family phrasing must also fail the advance gate.
    assert not is_agent_advance_intent(phrasing), phrasing


@pytest.mark.parametrize(
    "phrasing",
    [
        # Direct start phrases must stay recognized.
        "开始验证",
        "开始模型验证",
        "启动验证",
        "启动模型验证",
        "执行验证",
        "执行模型验证",
        "运行验证",
        "运行模型验证",
        "开始执行",
        "跑起来",
        # Bare affirmative commands.
        "开始",
        "启动",
        "start",
        "run",
        "validate",
        # 吧-suffixed affirmatives — the '吧$' collision decision must keep
        # these True (they are explicit direct_commands, not questions).
        "开始吧",
        "启动吧",
        "运行吧",
        "跑吧",
    ],
)
def test_start_intent_recognized(phrasing):
    assert is_start_validation_intent(phrasing), phrasing
    assert is_agent_advance_intent(phrasing), phrasing


@pytest.mark.parametrize(
    "phrasing",
    [
        # Negated starts: the substring branch would otherwise fire because
        # "开始验证" appears inside these strings.
        "先别开始验证",
        "先别开始模型验证",
        "不要开始验证",
        "不要开始模型验证",
        "别开始验证",
        "暂不开始验证",
        "不用开始验证",
        "不需要开始验证",
        "先不开始",
        # English negation parity with plan_driver._NEGATED_CONFIRM.
        "do not start validation",
        "don't run validation",
    ],
)
def test_start_negations_stay_chat(phrasing):
    assert not is_start_validation_intent(phrasing), phrasing
    assert not is_agent_advance_intent(phrasing), phrasing


@pytest.mark.parametrize(
    "phrasing",
    [
        # Interrogatives must route to chat, not launch a run.
        "什么时候开始验证?",
        "什么时候开始验证？",
        "什么时候开始模型验证?",
        "要不要开始验证",
        "要不要开始模型验证",
        "能不能开始验证",
        "开始验证吗？",
        "开始模型验证吗？",
        "现在可以开始验证吗",
    ],
)
def test_start_questions_stay_chat(phrasing):
    assert not is_start_validation_intent(phrasing), phrasing
    assert not is_agent_advance_intent(phrasing), phrasing
