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


class _SequencedLLM:
    def __init__(self, payloads: list[str]):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        if len(self.calls) <= len(self.payloads):
            return self.payloads[len(self.calls) - 1]
        return self.payloads[-1]


def test_parse_route_adjust_extracts_params():
    out = parse_route('{"action":"adjust","params":{"n_trials":20},"reason":"调大搜索"}')
    assert out["action"] == "adjust"
    assert out["params"] == {"n_trials": 20}


def test_parse_route_extracts_json_from_markdown():
    out = parse_route('模型判断如下:\n```json\n{"action":"adjust","params":{"n_trials":20},"reason":"调大搜索"}\n```')
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


def test_route_instruction_retries_once_after_unparseable_reply():
    fake = _SequencedLLM(["not json", '{"action":"adjust","params":{"n_trials":30},"reason":"重试可解析"}'])

    out = route_instruction(fake, gate_context="调参节点", instruction="n_trials 到 30")

    assert out["action"] == "adjust"
    assert out["params"] == {"n_trials": 30}
    assert len(fake.calls) == 2
    assert "上一次返回无法解析" in fake.calls[1]["user_prompt"]


def test_route_instruction_injects_param_schema_into_prompt():
    """AGT-5: the gate's adjustable-parameter schema (name/type/current value/
    bounds) is rendered into the prompt so the router extracts adjust params
    against real parameter names instead of guessing them from the instruction."""
    fake = _FakeLLM('{"action":"adjust","params":{"n_trials":30},"reason":"调大轮数"}')

    route_instruction(
        fake,
        gate_context="调参节点",
        instruction="n_trials 调到 30",
        param_schema=[
            {"name": "n_trials", "type": "integer", "current": 20, "bounds": {"min": 1}},
            {"name": "leakage_ks", "type": "number", "current": 0.4, "bounds": {"min": 0, "max": 1}},
        ],
    )

    prompt = fake.calls[0]["user_prompt"]
    assert "【可调参数】" in prompt
    assert "n_trials" in prompt
    assert "当前值=20" in prompt
    assert "leakage_ks" in prompt
    assert "min=0, max=1" in prompt


def test_route_instruction_omits_param_schema_section_when_empty():
    fake = _FakeLLM('{"action":"confirm","reason":"同意"}')

    route_instruction(fake, gate_context="计划总览", instruction="可以")

    assert "【可调参数】" not in fake.calls[0]["user_prompt"]
