from types import SimpleNamespace
import json

import pytest

from marvis.agent.turn_handlers.strategy_request import (
    _specialized_workflow_intake_route,
)
from tests.test_llm_outbound import Wire


def _task():
    return SimpleNamespace(
        task_type="modeling", target_type="binary", recipes=["lr"], sample_weight_col=""
    )


def _route():
    return {
        "action": "adjust",
        "params": {"target_type": "binary", "recipes": ["lr"], "n_trials": 1},
        "constraint": "",
        "reason": "抽取已声明初始规格",
        "confidence": "high",
        "explicit_authorization": False,
    }


def _review(instruction, **changes):
    return {
        "verdict": "authorize",
        "evidence_quote": instruction,
        "reason": "授权当前创建计划",
        "confidence": "high",
        "is_question": False,
        "is_conditional": False,
        "requests_change": False,
        "withholds_authorization": False,
        **changes,
    }


def test_initial_specs_need_independent_authorization_and_preserve_complete_protocol(
    monkeypatch,
):
    wire = Wire(monkeypatch)
    text = "请建立二分类模型，只用逻辑回归 lr，调参 1 轮；保留已有风险说明。"
    # Initial extraction has its own bounded completeness review; authorization
    # receives neither of the earlier decisions nor their reasons.
    wire.replies = [_route(), _route(), _review(text)]
    accepted, _ = _specialized_workflow_intake_route(
        SimpleNamespace(llm_client=wire.client), _task(), instruction=text
    )
    assert accepted["params"] == _route()["params"]
    assert len(wire.requests) == 3
    first = json.loads(wire.requests[0]["messages"][1]["content"])
    assert first["instruction"] == text
    assert first["gate_context"]["purpose"] == "create_workflow_plan"
    assert first["gate_context"]["binding"]["workflow_family"] == "modeling"
    assert "即使等于当前默认值" in wire.requests[0]["messages"][0]["content"]
    final = wire.last_context()
    assert (
        final["instruction"] == text and final["proposed_params"] == _route()["params"]
    )
    assert final["gate_context"]["purpose"] == "create_workflow_plan"
    assert final["gate_context"]["binding"]["workflow_family"] == "modeling"
    assert "first_pass" not in final and "review_instruction" not in final


@pytest.mark.parametrize(
    ("text", "flags"),
    [
        (
            "是不是应该建立模型，使用 lr？",
            {"verdict": "ambiguous", "is_question": True},
        ),
        ("如果核验通过再用 lr 建模。", {"is_conditional": True}),
        ("不要用 lr 建模。", {"verdict": "reject", "withholds_authorization": True}),
        (
            "lr 看起来可以，再考虑一下。",
            {"verdict": "ambiguous", "withholds_authorization": True},
        ),
        ("先换成 xgb，然后建模。", {"requests_change": True}),
        ("建立模型。", {"confidence": "medium"}),
    ],
)
def test_false_positive_first_extraction_cannot_bypass_independent_review(
    monkeypatch, text, flags
):
    wire = Wire(monkeypatch)
    wire.replies = [_route(), _route(), _review(text, **flags)]
    accepted, _ = _specialized_workflow_intake_route(
        SimpleNamespace(llm_client=wire.client), _task(), instruction=text
    )
    assert accepted is None
    assert wire.last_context()["instruction"] == text
    assert wire.requests[-1]["messages"][0]["content"].startswith(
        "你是 MARVIS 确认节点的第二遍独立"
    )
