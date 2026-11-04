"""Strategy-development clarifications resume from the user's answers (eval S8).

The eval dead loop: the compiler asked "审批还是拒绝", the answer was refused
by top-level routing, a rephrase got a raw ``objective=max_approval`` demand,
and the next answer re-asked the first question.  These tests pin that answers
are routed back to the pending question, merged across turns, read in business
Chinese, and that the home-extracted task contract is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marvis.agent.quick_replies import QUICK_REPLIES_KEY
from marvis.agent.strategy_request_compiler.development import (
    is_strategy_development_request,
    parse_development_slots,
)
from marvis.app import create_app
from tests.strategy_sample_design_support import (
    confirm_strategy_score_direction,
    materialize_mature_strategy_sample_design,
)

_ASKS_TYPE = "请明确策略类型是审批（approval）还是拒绝（reject），并说明是否已有基线策略或指定策略 ID"


class _ClarifyingLLM:
    """A compiler model that never manages to settle the request itself."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return json.dumps({"clarification": _ASKS_TYPE}, ensure_ascii=False)


def _install(monkeypatch, llm) -> None:
    monkeypatch.setattr(
        "marvis.agent.validation_app_service.driver_llm_client",
        lambda request, task: llm,
    )
    monkeypatch.setattr(
        "marvis.routers.validation_agent.resolve_driver_agent_client",
        lambda _request, _task, _payload: llm,
    )


def _task(
    client: TestClient,
    tmp_path: Path,
    *,
    strategy_input: dict | None = None,
    run_mode: str = "manual",
) -> str:
    source = tmp_path / "strategy-source"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 4, 5],
            "bad": [0, 0, 1, 0, 1, 1],
            "score": [800, 760, 720, 680, 640, 600],
        }
    ).to_csv(source / "sample.csv", index=False)
    body = {
        "model_name": "一套贷前准入策略",
        "validator": "qa",
        "source_dir": str(source),
        "task_type": "strategy",
        "run_mode": run_mode,
        "target_col": "bad",
        "score_col": "score",
    }
    if strategy_input is not None:
        body["strategy_input"] = strategy_input
    response = client.post("/api/tasks", json=body)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def _send(client: TestClient, task_id: str, content: str) -> dict:
    response = client.post(
        f"/api/tasks/{task_id}/agent/messages", json={"content": content}
    )
    assert response.status_code == 202, response.text
    return response.json()


def _plans(client: TestClient, task_id: str) -> list[dict]:
    return [
        plan
        for plan in client.get(f"/api/tasks/{task_id}/plans").json()["plans"]
        if plan["template_id"] == "strategy_development"
    ]


def _last_assistant(payload: dict) -> dict:
    return [m for m in payload["messages"] if m["role"] == "assistant"][-1]


def test_slot_parser_reads_eval_answers_without_defaults() -> None:
    assert parse_development_slots("准入就是审批策略，没有基线，从零开始做") == {
        "strategy_type": "approval",
        "baseline": "none",
    }
    assert parse_development_slots(
        "目标是在坏账不超过 5% 的前提下尽量提高通过率"
    ) == {"objective": "max_approval", "max_bad_rate": 0.05}
    assert parse_development_slots("尽量提高收益") == {"objective": "max_profit"}
    assert parse_development_slots("不是拒绝，是审批")["strategy_type"] == "approval"
    # "拒绝率" is a rate, not a strategy type; an open question picks nothing.
    assert "strategy_type" not in parse_development_slots("拒绝率不超过 40%")
    assert parse_development_slots("是审批还是拒绝？") == {}
    assert is_strategy_development_request("用这份样本做一套贷前准入策略")
    assert not is_strategy_development_request("回测已有审批策略")


def test_home_contract_and_utterance_start_development_without_asking(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(
        client,
        tmp_path,
        strategy_input={
            "entry_mode": "strategy_development",
            "objective": "max_approval",
            "max_bad_rate": 0.05,
            "min_approval_rate": 0.6,
        },
    )
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)
    llm = _ClarifyingLLM()
    _install(monkeypatch, llm)

    _send(
        client,
        task_id,
        "开始策略开发。我的目标：用这份样本做一套贷前准入策略，score 越高风险越高，"
        "通过率不低于 60%，通过客群坏账控制在 5% 以内",
    )

    plans = _plans(client, task_id)
    assert len(plans) == 1
    assert plans[0]["success_criteria"] == [
        {"metric": "approved_bad_rate", "max": 0.05},
        {"metric": "approval_rate", "min": 0.6},
    ]
    contract = client.get(f"/api/tasks/{task_id}").json()["strategy_input"]
    assert contract["strategy_type"] == "approval"
    assert contract["objective"] == "max_approval"


def test_answers_merge_across_turns_and_resume_the_same_request(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    confirm_strategy_score_direction(client, task_id, direction="lower_is_riskier")
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)
    llm = _ClarifyingLLM()
    _install(monkeypatch, llm)

    first = _send(client, task_id, "帮我开发一套策略，通过率不低于 60%")
    question = _last_assistant(first)
    assert first["status"] == "clarification_required"
    assert first["code"] == "strategy_development_inputs_required"
    assert "审批（准入）策略，还是拒绝策略" in question["content"]
    assert "通过率不低于 60%" in question["content"]
    for internal in ("approval", "reject", "objective", "max_approval", "策略 ID"):
        assert internal not in question["content"]
    assert [reply["command"] for reply in question["metadata"][QUICK_REPLIES_KEY]] == [
        "审批策略",
        "拒绝策略",
    ]
    assert _plans(client, task_id) == []
    compiler_calls = len(llm.calls)

    second = _send(client, task_id, "准入就是审批策略，没有基线，从零开始做")
    asked = _last_assistant(second)
    assert second["code"] == "strategy_development_inputs_required"
    assert "尽量提高通过率，还是尽量提高收益" in asked["content"]
    assert "审批（准入）策略" in asked["content"]  # first answer remembered
    assert [reply["command"] for reply in asked["metadata"][QUICK_REPLIES_KEY]] == [
        "尽量提高通过率",
        "尽量提高收益",
    ]
    assert _plans(client, task_id) == []

    # The chip command is exactly what the answer handler accepts.
    _send(client, task_id, "尽量提高通过率")
    plans = _plans(client, task_id)
    assert len(plans) == 1
    assert plans[0]["success_criteria"] == [{"metric": "approval_rate", "min": 0.6}]
    # Answers never went back through the compiler model.
    assert len(llm.calls) == compiler_calls


def test_question_hold_back_and_cancel_never_start_a_plan(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)
    _install(monkeypatch, _ClarifyingLLM())

    _send(client, task_id, "开发一套审批策略，坏账不超过 5%")

    explained = _last_assistant(_send(client, task_id, "两种目标有什么区别？"))
    assert "尽量提高通过率" in explained["content"]
    assert "还没有创建计划" in explained["content"]
    held = _last_assistant(
        _send(client, task_id, "尽量提高通过率，但先别开始，我再确认下样本")
    )
    assert "先不开始" in held["content"]
    assert [reply["command"] for reply in held["metadata"][QUICK_REPLIES_KEY]] == [
        "开始策略开发"
    ]
    assert _plans(client, task_id) == []

    cancelled = _last_assistant(_send(client, task_id, "算了"))
    assert cancelled["metadata"]["intent"] == "strategy_request_cancelled"
    assert _plans(client, task_id) == []


def test_withheld_answers_start_on_the_offered_start_command(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    confirm_strategy_score_direction(client, task_id, direction="lower_is_riskier")
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)
    _install(monkeypatch, _ClarifyingLLM())

    _send(client, task_id, "开发一套审批策略，坏账不超过 5%")
    _send(client, task_id, "尽量提高通过率，但先别开始")
    assert _plans(client, task_id) == []
    _send(client, task_id, "开始策略开发")
    assert len(_plans(client, task_id)) == 1


def test_agent_mode_answer_is_not_sent_back_to_top_level_routing(
    tmp_path: Path, monkeypatch
) -> None:
    class _AgentLLM:
        def __init__(self) -> None:
            self.callers: list[str] = []

        def complete(self, **kwargs):
            caller = str(kwargs.get("caller") or "")
            self.callers.append(caller)
            if caller.startswith("semantic_intent"):
                request = json.loads(kwargs["user_prompt"])
                return json.dumps(
                    {
                        "intent": "strategy_workflow",
                        "evidence_quote": request["instruction"],
                        "reason": "用户要求开发策略。",
                        "confidence": "high",
                        "is_question": False,
                        "is_conditional": False,
                        "requests_change": False,
                        "withholds_action": False,
                    },
                    ensure_ascii=False,
                )
            return json.dumps({"clarification": _ASKS_TYPE}, ensure_ascii=False)

    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path, run_mode="agent")
    confirm_strategy_score_direction(client, task_id, direction="lower_is_riskier")
    materialize_mature_strategy_sample_design(client, task_id, monkeypatch)
    llm = _AgentLLM()
    _install(monkeypatch, llm)

    _send(
        client,
        task_id,
        "开始策略开发。我的目标：做一套策略，坏账不超过 5% 的前提下尽量提高通过率",
    )
    assert "semantic_intent_router" in llm.callers
    before = list(llm.callers)

    answered = _send(client, task_id, "准入就是审批策略，没有基线，从零开始做")

    assert llm.callers == before
    assert all(
        (m.get("metadata") or {}).get("code") != "semantic_intent_clarification"
        for m in answered["messages"]
    )
    assert len(_plans(client, task_id)) == 1


@pytest.mark.parametrize(
    "content",
    ["准入就是审批策略", "尽量提高通过率"],
)
def test_bare_answers_without_a_pending_question_keep_normal_routing(
    tmp_path: Path, monkeypatch, content: str
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _ClarifyingLLM()
    _install(monkeypatch, llm)

    payload = _send(client, task_id, content)

    assert payload["status"] == "clarification_required"
    assert _plans(client, task_id) == []
    assert all(
        "strategy_development_clarification" not in (m.get("metadata") or {})
        for m in payload["messages"]
    )


def test_a_different_strategy_operation_is_not_taken_as_an_answer(
    tmp_path: Path, monkeypatch
) -> None:
    client = TestClient(create_app(tmp_path))
    task_id = _task(client, tmp_path)
    llm = _ClarifyingLLM()
    _install(monkeypatch, llm)

    _send(client, task_id, "帮我开发一套策略，通过率不低于 60%")
    compiler_calls = len(llm.calls)
    payload = _send(client, task_id, "回测已有审批策略")

    # Normal routing compiled it as a new request instead of merging slots.
    assert len(llm.calls) > compiler_calls
    assert _last_assistant(payload)["metadata"].get("code") != (
        "strategy_development_inputs_required"
    )
    assert _plans(client, task_id) == []
