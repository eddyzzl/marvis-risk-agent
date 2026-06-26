"""Agent-mode gate-instruction router: classify a free-text reply at a gate into
confirm / adjust / replan / clarify and extract parameters. Pure + FakeLLM-driven.
"""

from __future__ import annotations

from marvis.agent.instruction_router import parse_route, route_instruction


class _FakeLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.payload


def test_parse_route_adjust_extracts_params():
    out = parse_route('{"action":"adjust","params":{"n_trials":20},"reason":"调大搜索"}')
    assert out["action"] == "adjust"
    assert out["params"] == {"n_trials": 20}


def test_parse_route_adjust_without_params_falls_back_to_clarify():
    out = parse_route('{"action":"adjust","params":{},"reason":""}')
    assert out["action"] == "clarify"
    assert out["reason"]


def test_parse_route_replan_keeps_constraint():
    out = parse_route('{"action":"replan","constraint":"改用 xgb 重新建模","reason":"x"}')
    assert out["action"] == "replan"
    assert out["constraint"] == "改用 xgb 重新建模"


def test_parse_route_junk_is_clarify():
    assert parse_route("not json at all")["action"] == "clarify"


def test_parse_route_unknown_action_is_clarify():
    assert parse_route('{"action":"frobnicate","params":{}}')["action"] == "clarify"


def test_route_instruction_passes_context_and_instruction_to_llm():
    fake = _FakeLLM('{"action":"confirm","reason":"同意"}')
    out = route_instruction(fake, gate_context="特征筛选完成", instruction="可以,继续")
    assert out["action"] == "confirm"
    prompt = fake.calls[0]["user_prompt"]
    assert "特征筛选完成" in prompt
    assert "可以,继续" in prompt
