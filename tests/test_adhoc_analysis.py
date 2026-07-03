"""S6 Commit 1: ad-hoc question-intent parsing + 口径确认门.

The LLM only produces a structured spec (never a number, INV-1); the platform
validates every column against the dataset profile whitelist and rejects any
hallucinated column with a Chinese clarification (never a guess). A validated spec
carries a 口径确认门 text -- the caller must confirm before slice_aggregate runs.
"""

from __future__ import annotations

import json

from marvis.agent.adhoc_analysis import (
    build_slice_spec_from_utterance,
    detect_question_intent,
    validate_slice_spec,
)


class _FakeLLM:
    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def complete(self, *, system_prompt, user_prompt, **kwargs):
        self.calls += 1
        return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)


_PROFILE = ["channel", "month", "bad", "decision", "amount"]


def test_detect_question_intent_conservative():
    assert detect_question_intent("按渠道看 5 月坏率") is True
    assert detect_question_intent("统计各渠道通过率") is True
    # A non-question turn defaults back to the normal flow (returns False).
    assert detect_question_intent("确认") is False
    assert detect_question_intent("") is False
    assert detect_question_intent(None) is False


def test_build_spec_produces_confirmation_gate_first():
    llm = _FakeLLM({
        "group_by": ["channel"],
        "metrics": [{"op": "bad_rate", "col": "bad"}],
        "month_col": "month",
        "months": ["2026-05"],
    })

    result = build_slice_spec_from_utterance("按渠道看 5 月坏率", _PROFILE, llm)

    assert result.needs_clarification is False
    assert result.spec is not None
    # 口径确认门先行: a plain-Chinese echo of exactly what will run, ending in 确认？
    assert "将按〔channel〕统计〔bad 的坏率〕" in result.confirmation_text
    assert "2026-05" in result.confirmation_text
    assert result.confirmation_text.endswith("，确认？")
    # The single-step plan inputs echo the口径 unchanged.
    inputs = result.spec.tool_inputs("ds-1")
    assert inputs["group_by"] == ["channel"]
    assert inputs["metrics"] == [{"op": "bad_rate", "col": "bad"}]
    assert inputs["month_col"] == "month"
    assert inputs["months"] == ["2026-05"]


def test_hallucinated_column_yields_chinese_clarification_not_a_guess():
    llm = _FakeLLM({
        "group_by": ["region"],  # not in the profile whitelist
        "metrics": [{"op": "count"}],
    })

    result = build_slice_spec_from_utterance("按地区看数量", _PROFILE, llm)

    assert result.needs_clarification is True
    assert result.spec is None
    assert "region" in result.clarify
    assert result.confirmation_text is None


def test_hallucinated_metric_column_rejected():
    result = validate_slice_spec(
        {"metrics": [{"op": "mean", "col": "ghost_col"}]},
        _PROFILE,
    )
    assert result.needs_clarification is True
    assert "ghost_col" in result.clarify


def test_unsupported_op_rejected():
    result = validate_slice_spec(
        {"metrics": [{"op": "median", "col": "amount"}]},
        _PROFILE,
    )
    assert result.needs_clarification is True
    assert "median" in result.clarify


def test_llm_clarify_passthrough_when_intent_unclear():
    llm = _FakeLLM({"clarify": "请问你想看哪个指标？"})

    result = build_slice_spec_from_utterance("那个东西怎么样", _PROFILE, llm)

    assert result.needs_clarification is True
    assert result.clarify == "请问你想看哪个指标？"


def test_parse_failure_yields_clarification():
    llm = _FakeLLM("not json at all")

    result = build_slice_spec_from_utterance("看一下坏率", _PROFILE, llm)

    assert result.needs_clarification is True
    assert result.clarify


def test_valid_spec_without_group_by_uses_whole_population_text():
    result = validate_slice_spec(
        {"metrics": [{"op": "count"}]},
        _PROFILE,
    )
    assert result.needs_clarification is False
    assert "全体样本" in result.confirmation_text
