"""Plan critique and business review receive the user's goal verbatim.

Negations, conditions and thresholds must survive to the actual wire body;
a goal holding disallowed material is left out of the request entirely.
"""

import json

import pytest

from marvis.orchestrator.reviewer import Reviewer
from tests.test_llm_outbound import Wire
from tests.test_orch_reviewer import _plan, _step


GOALS = [
    "训练一个LR模型，不要用XGB，KS>0.3",
    "如果OOT KS不低于0.35且PSI小于0.1才算通过；不要删除age字段，样本权重列设为w",
    "Build an LR scorecard, do not use XGB unless test AUC < 0.7",
]
DISALLOWED = [
    "训练LR模型，api_key=TEST_ONLY_VALUE",
    "读取 /Users/example/private/data.csv 后训练LR，KS>0.3",
    "客户姓名：测试姓名，训练LR模型",
    "访问 https://private.invalid/data 再建模",
]


def _wire_context(wire):
    return json.loads(wire.requests[-1]["messages"][1]["content"])


@pytest.mark.parametrize("goal", GOALS)
def test_critique_wire_body_carries_goal_verbatim(monkeypatch, goal):
    wire = Wire(monkeypatch)
    wire.reply = {"passed": True, "reasons": []}
    verdict = Reviewer(lambda: wire.client).llm_critique(_step([]), {"oot_ks": .31}, goal)
    assert verdict.passed is True
    assert len(wire.requests) == 1
    context = _wire_context(wire)
    assert context["goal"] == goal
    assert "request_topics" not in context


@pytest.mark.parametrize("goal", GOALS)
def test_business_review_wire_body_carries_goal_verbatim(monkeypatch, goal):
    wire = Wire(monkeypatch)
    wire.reply = {"summary": "依据平台验收结果解释", "goal_met": None}
    review = Reviewer(lambda: wire.client).final_review(
        _plan(_step([])), {"step-1": {"metrics": {"oot_ks": .31}}}, goal)
    assert review.summary == "依据平台验收结果解释"
    assert len(wire.requests) == 1
    context = _wire_context(wire)
    assert context["goal"] == goal
    assert "request_topics" not in context
    assert context["business_acceptance"]["status"] == review.business_acceptance["status"]


_SECRETS = ("TEST_ONLY_VALUE", "/Users/example", "测试姓名", "private.invalid")


@pytest.mark.parametrize("goal", DISALLOWED)
def test_critique_omits_a_refused_goal_and_still_reviews(monkeypatch, goal):
    # Review B5: a refused goal is left out; it never becomes a critic warning.
    wire = Wire(monkeypatch)
    wire.reply = {"passed": True, "reasons": []}
    verdict = Reviewer(lambda: wire.client).llm_critique(_step([]), {"oot_ks": .31}, goal)
    assert verdict.passed is True and verdict.status != "failed"
    assert len(wire.requests) == 1
    context = _wire_context(wire)
    assert "goal" not in context and "request_topics" not in context
    body = json.dumps(wire.requests, ensure_ascii=False)
    for secret in _SECRETS:
        assert secret not in body


@pytest.mark.parametrize("goal", DISALLOWED)
def test_business_review_omits_a_refused_goal_and_still_explains(monkeypatch, goal):
    wire = Wire(monkeypatch)
    wire.reply = {"summary": "依据平台验收结果解释", "goal_met": None}
    review = Reviewer(lambda: wire.client).final_review(
        _plan(_step([])), {"step-1": {"metrics": {"oot_ks": .31}}}, goal)
    assert review.summary == "依据平台验收结果解释"
    assert len(wire.requests) == 1
    assert "goal" not in _wire_context(wire)
    body = json.dumps(wire.requests, ensure_ascii=False)
    for secret in _SECRETS:
        assert secret not in body
    # The deterministic business verdict is untouched.
    assert review.business_acceptance["status"] == "not_configured"


@pytest.mark.parametrize("goal", [None, "", "   "])
def test_missing_goal_is_omitted_not_rewritten(monkeypatch, goal):
    wire = Wire(monkeypatch)
    wire.reply = {"passed": True, "reasons": []}
    Reviewer(lambda: wire.client).llm_critique(_step([]), {"oot_ks": .31}, goal)
    context = _wire_context(wire)
    assert "goal" not in context
    assert "request_topics" not in context
