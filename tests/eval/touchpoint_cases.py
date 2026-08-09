"""Test-only degraded-output corpus for weak-model JSON touchpoints (TST-1).

The plan-level framework in ``cases.py``/``runner.py`` (LLM-2) exercises
IntentRouter/Planner/Validator end-to-end against realistic *well-formed*
plan JSON. This module is narrower and orthogonal: it targets the exact
degraded raw-text shapes a weak local model actually emits at each of the
four structured-JSON touchpoints --

  - decide_gate      (marvis/agent/auto_drive.py)
  - route_instruction (marvis/agent/instruction_router.py)
  - planner.generate / .replan / .next_explore_segment (marvis/orchestrator/planner.py)
  - reviewer soft critique (marvis/orchestrator/reviewer.py: llm_critique / final_review)

For each touchpoint there are at least six degradation categories:

  1. markdown fence      - ```json ... ``` wrapping
  2. prose prefix/suffix  - chatty preamble/postscript around the JSON
  3. key casing / quoting - wrong-case keys or single-quoted "JSON"
  4. truncated JSON       - the reply is cut off mid-object
  5. negation semantics   - natural-language "don't confirm" contradicting
                            (or replacing) the structured action field
  6. <think> contamination - reasoning-model draft JSON mixed with the
                             final answer

Each ``TouchpointCase`` records the *actual, currently observed* outcome
when driven through the real production function (not a re-implementation)
so this file is both a regression lock for the safe paths and an honest,
executable record of the paths that are not yet safe (``expected_failure``).

Canned parser outputs intentionally live under ``tests``: they must not be
counted as production model-planning or guardrail evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TouchpointCase:
    id: str
    touchpoint: str  # "decide_gate" | "route_instruction" | "planner" | "reviewer"
    degradation: str  # one of the six categories described above
    # A callable of (raw_or_sequence) -> actual_result, wired up per touchpoint
    # by the test module (kept out of this fixture file so this module stays
    # a pure data table with no import-time dependency on the touchpoints).
    raw_output: Any
    # Structural predicate the harness must confirm against the touchpoint's
    # actual return value: dict of expected key -> expected value, checked
    # with equality (missing keys in actual are ignored).
    expected: dict[str, Any]
    # When set, this case documents a currently-real, confirmed-unsafe
    # behavior (see docstrings below / final report) rather than a behavior
    # the platform actually guarantees today. The regression test asserts
    # the *documented* (unsafe) outcome so a silent behavior change is
    # caught either way -- fixing it should update/remove this marker, not
    # silently keep the test green.
    expected_failure: str = ""
    notes: str = ""


# -- decide_gate ---------------------------------------------------------
#
# marvis/agent/auto_drive.py:decide_gate wraps _parse_decision (JSON
# extraction via marvis.agent.json_reply.load_json_object) with one retry on
# unparseable replies, then a deterministic AUTO safety policy
# (_apply_safety_policy) that can still force a halt even for a "valid"
# decision. Fallback on total parse failure is always {"action": "halt"}.

DECIDE_GATE_CASES: tuple[TouchpointCase, ...] = (
    TouchpointCase(
        id="decide_gate_fence",
        touchpoint="decide_gate",
        degradation="markdown_fence",
        raw_output='```json\n{"action":"confirm","reason":"指标稳定"}\n```',
        expected={"action": "confirm", "reason": "指标稳定"},
        notes="load_json_object strips the fence; safe pass-through, no retry needed.",
    ),
    TouchpointCase(
        id="decide_gate_prose_prefix_suffix",
        touchpoint="decide_gate",
        degradation="prose_prefix_suffix",
        raw_output="根据结果分析如下:\n"
        '{"action":"confirm","reason":"指标正常"}\n'
        "以上是我的判断,请参考。",
        expected={"action": "confirm", "reason": "指标正常"},
        notes="_extract_first_object finds the embedded object; safe.",
    ),
    TouchpointCase(
        id="decide_gate_key_casing",
        touchpoint="decide_gate",
        degradation="key_casing",
        raw_output='{"Action":"CONFIRM","Reason":"ok"}',
        expected={"action": "halt"},
        notes=(
            "data.get('action') is exact-case; 'Action' is never read, action "
            "defaults to empty -> unknown action -> safe halt. Single-shot "
            "parse_decision does not retry; decide_gate's own retry wrapper "
            "would burn one extra round-trip here before still halting."
        ),
    ),
    TouchpointCase(
        id="decide_gate_truncated_json",
        touchpoint="decide_gate",
        degradation="truncated_json",
        raw_output='{"action":"conf',
        expected={"action": "halt", "reason": "无法解析模型决策，转人工确认。"},
        notes="Unparseable -> safe halt fallback; decide_gate's retry gets a fresh chance.",
    ),
    TouchpointCase(
        id="decide_gate_negation_in_reason_contradicts_action",
        touchpoint="decide_gate",
        degradation="negation_semantics",
        raw_output=(
            '{"action":"confirm","reason":"指标异常，不能确认，不可以继续。"}'
        ),
        expected={"action": "halt"},
        notes=(
            "A confirm action whose own reason explicitly rejects confirmation "
            "must fail closed to human review."
        ),
    ),
    TouchpointCase(
        id="decide_gate_think_tag_draft_vs_final_reversed",
        touchpoint="decide_gate",
        degradation="think_tag_contamination",
        raw_output=(
            "<think>看起来正常，先给 "
            '{"action":"confirm","reason":"看起来正常"} '
            "但仔细看命中率异常偏低，"
            "应该停下</think>最终 "
            '{"action":"halt","reason":"命中率异常偏低，需人工复核"}'
        ),
        # LLM-6 (strip_thinking_segments at the client boundary) removes the
        # <think> draft before JSON extraction, so the FINAL halt decision wins.
        # This case originally documented the draft-vs-final reversal as an
        # expected failure; it now guards the fix against regression.
        expected={"action": "halt", "reason": "命中率异常偏低，需人工复核"},
        expected_failure=None,
    ),
)


# -- route_instruction -----------------------------------------------------
#
# marvis/agent/instruction_router.py:route_instruction. Same
# load_json_object + one retry pattern as decide_gate; safe fallback on
# total parse failure is {"action": "clarify"}.

ROUTE_INSTRUCTION_CASES: tuple[TouchpointCase, ...] = (
    TouchpointCase(
        id="route_instruction_fence",
        touchpoint="route_instruction",
        degradation="markdown_fence",
        raw_output='模型判断如下:\n```json\n{"action":"adjust","params":{"n_trials":20},"reason":"调大搜索"}\n```',
        expected={"action": "adjust", "params": {"n_trials": 20}},
        notes="Already covered by tests/test_instruction_router.py; included here for matrix completeness.",
    ),
    TouchpointCase(
        id="route_instruction_prose_prefix_suffix",
        touchpoint="route_instruction",
        degradation="prose_prefix_suffix",
        raw_output=(
            "根据用户指令,我的判断是:\n"
            '{"action":"replan","constraint":"换用 xgb 重新建模","reason":"结构性改动"}\n'
            "希望这个判断对你有帮助。"
        ),
        expected={"action": "replan", "constraint": "换用 xgb 重新建模"},
        notes="_extract_first_object recovers the embedded object; safe.",
    ),
    TouchpointCase(
        id="route_instruction_key_casing",
        touchpoint="route_instruction",
        degradation="key_casing",
        raw_output='{"ACTION":"adjust","Params":{"n_trials":20},"Reason":"x"}',
        expected={"action": "clarify"},
        notes="Wrong-case keys are never read; action defaults to '' -> clarify. Safe.",
    ),
    TouchpointCase(
        id="route_instruction_truncated_json",
        touchpoint="route_instruction",
        degradation="truncated_json",
        raw_output='{"action":"adjust","params":{"n_trials":2',
        expected={"action": "clarify"},
        notes="Unparseable -> safe clarify fallback; route_instruction's retry gets a fresh chance.",
    ),
    TouchpointCase(
        id="route_instruction_negation_in_reason_contradicts_action",
        touchpoint="route_instruction",
        degradation="negation_semantics",
        raw_output=(
            '{"action":"confirm","reason":"用户其实是说不同意,不要继续。"}'
        ),
        expected={"action": "clarify"},
        notes=(
            "A confirm action whose own reason says the user disagreed and "
            "must not continue is downgraded to clarification."
        ),
    ),
    TouchpointCase(
        id="route_instruction_think_tag_draft_vs_final_reversed",
        touchpoint="route_instruction",
        degradation="think_tag_contamination",
        raw_output=(
            '<think>用户说"改成两周",草稿 '
            '{"action":"confirm"} 不对,是想调整参数'
            "</think>"
            '{"action":"adjust","params":{"horizon_weeks":2},"reason":"改成两周"}'
        ),
        # LLM-6 strips the <think> draft before extraction: the final adjust
        # decision (the user's real instruction) wins. Regression guard for the fix.
        expected={"action": "adjust", "params": {"horizon_weeks": 2}},
        expected_failure=None,
    ),
)


# -- planner (generate / replan / next_explore_segment) --------------------
#
# Planner parsing now shares load_json_object with the other LLM touchpoints.
# Fence/prose/think cases therefore use a valid non-empty plan and must succeed
# in one call. Truncated JSON and wrong-case schema keys remain fail-closed
# PlanningError cases; that rejection is expected safety behavior, not an
# expected_failure ledger entry.

_VALID_ECHO_PLAN = (
    '{"steps":[{"id":"step-1","title":"echo",'
    '"tool":{"plugin":"_sample","tool":"echo"},'
    '"inputs":{"message":"ok"},"depends_on":[],"post_checks":[]}]}'
)

PLANNER_CASES: tuple[TouchpointCase, ...] = (
    TouchpointCase(
        id="planner_fence_valid_plan",
        touchpoint="planner",
        degradation="markdown_fence",
        raw_output=f"```json\n{_VALID_ECHO_PLAN}\n```",
        expected={"call_count": 1, "tools": ["_sample.echo"], "step_count": 1},
        notes="Fenced valid plan is extracted and validated in one call.",
    ),
    TouchpointCase(
        id="planner_prose_prefix_suffix_valid_plan",
        touchpoint="planner",
        degradation="prose_prefix_suffix",
        raw_output=f"计划如下：\n{_VALID_ECHO_PLAN}\n以上。",
        expected={"call_count": 1, "tools": ["_sample.echo"], "step_count": 1},
        notes="A valid embedded plan is extracted and validated in one call.",
    ),
    TouchpointCase(
        id="planner_truncated_json_exhausts_retries",
        touchpoint="planner",
        degradation="truncated_json",
        raw_output='{"steps": [{"title": "a"',
        expected={"raises": "PlanningError", "call_count": 3},
        notes=(
            "A truncated plan is rejected after the bounded retry budget; "
            "partial execution is intentionally forbidden."
        ),
    ),
    TouchpointCase(
        id="planner_think_tag_valid_plan",
        touchpoint="planner",
        degradation="think_tag_contamination",
        raw_output=(
            "<think>这个任务需要先读取数据"
            "</think>"
            f"{_VALID_ECHO_PLAN}"
        ),
        expected={"call_count": 1, "tools": ["_sample.echo"], "step_count": 1},
        notes="Thinking text is stripped and the final valid plan is used.",
    ),
    TouchpointCase(
        id="planner_key_casing_is_schema_error_not_parse_error",
        touchpoint="planner",
        degradation="key_casing",
        raw_output='{"Steps": []}',
        expected={"raises": "PlanningError", "call_count": 3},
        notes=(
            "The object parses, but wrong-case Steps does not satisfy the "
            "planner schema and is rejected after the bounded retry budget."
        ),
    ),
    TouchpointCase(
        id="planner_negation_semantics_not_applicable",
        touchpoint="planner",
        degradation="negation_semantics",
        raw_output=_VALID_ECHO_PLAN,
        expected={"call_count": 1, "tools": ["_sample.echo"], "step_count": 1},
        notes=(
            "The planner has no confirm/halt action field -- negation "
            "semantics is a decide_gate/route_instruction concept. This valid "
            "plan keeps the matrix category explicit without manufacturing an "
            "unrelated empty-plan failure."
        ),
    ),
)


# -- reviewer (llm_critique soft verdict + final_review summarize) ---------
#
# marvis/orchestrator/reviewer.py uses load_json_object (same as
# decide_gate/route_instruction) with one retry via _retry_json_prompt.
# Unlike decide_gate/route_instruction, a totally unparseable reply after
# retry becomes passed=False (llm_critique) or a neutral "Plan execution
# reviewed." summary with llm_goal_met=None (final_review) -- both are safe
# in the sense that they never silently mark a plan/step as passing when the
# reviewer text made no sense, and per AGT-3, an LLM verdict alone can never
# FAIL a plan outright (only mark REVIEW/doubt), which bounds the blast
# radius of the same <think> extraction bug seen in decide_gate.

REVIEWER_CASES: tuple[TouchpointCase, ...] = (
    TouchpointCase(
        id="reviewer_critique_fence",
        touchpoint="reviewer",
        degradation="markdown_fence",
        raw_output='```json\n{"passed": true, "reasons": []}\n```',
        expected={"passed": True, "reasons": []},
        notes="load_json_object strips the fence; safe pass-through.",
    ),
    TouchpointCase(
        id="reviewer_critique_prose_prefix_suffix",
        touchpoint="reviewer",
        degradation="prose_prefix_suffix",
        raw_output=(
            "我的评审结果如下:\n"
            '{"passed": false, "reasons": ["oot_ks 偏低"]}\n'
            "请参考。"
        ),
        expected={"passed": False, "reasons": ["oot_ks 偏低"]},
        notes="_extract_first_object recovers the embedded object; safe.",
    ),
    TouchpointCase(
        id="reviewer_critique_key_casing",
        touchpoint="reviewer",
        degradation="key_casing",
        raw_output='{"Passed": true, "Reasons": []}',
        expected={"passed": False, "reasons": []},
        notes=(
            "A missing exact-case passed boolean fails closed instead of "
            "defaulting an unrecognized payload to success."
        ),
    ),
    TouchpointCase(
        id="reviewer_critique_truncated_json",
        touchpoint="reviewer",
        degradation="truncated_json",
        raw_output='{"passed": fal',
        expected={"passed": False, "reasons": ["llm critique returned non-json"]},
        notes="Unparseable -> safe passed=False with an explicit reason; retry gets a fresh chance.",
    ),
    TouchpointCase(
        id="reviewer_critique_negation_in_reasons_contradicts_passed",
        touchpoint="reviewer",
        degradation="negation_semantics",
        raw_output=(
            '{"passed": true, "reasons": ["指标异常，不应该通过，建议重新训练"]}'
        ),
        expected={"passed": False},
        notes=(
            "A positive verdict whose own reasons explicitly say it should "
            "not pass is downgraded to a failed soft review."
        ),
    ),
    TouchpointCase(
        id="reviewer_critique_think_tag_draft_vs_final_reversed",
        touchpoint="reviewer",
        degradation="think_tag_contamination",
        raw_output=(
            "<think>草稿觉得 "
            '{"passed": true, "reasons": []} '
            "但仔细看有问题</think>"
            '{"passed": false, "reasons": ["oot_ks 明显低于 train_ks,疑似过拟合"]}'
        ),
        # LLM-6 strips the <think> draft before extraction: the model's final
        # verdict (passed=false, overfitting flag) wins. Regression guard.
        expected={"passed": False, "reasons": ["oot_ks 明显低于 train_ks,疑似过拟合"]},
        expected_failure=None,
    ),
)


ALL_TOUCHPOINT_CASES: tuple[TouchpointCase, ...] = (
    DECIDE_GATE_CASES
    + ROUTE_INSTRUCTION_CASES
    + PLANNER_CASES
    + REVIEWER_CASES
)


def cases_by_touchpoint(
    cases: tuple[TouchpointCase, ...] | None = None,
) -> dict[str, tuple[TouchpointCase, ...]]:
    source = cases if cases is not None else ALL_TOUCHPOINT_CASES
    grouped: dict[str, list[TouchpointCase]] = {}
    for case in source:
        grouped.setdefault(case.touchpoint, []).append(case)
    return {touchpoint: tuple(items) for touchpoint, items in grouped.items()}


def cases_by_degradation(
    cases: tuple[TouchpointCase, ...] | None = None,
) -> dict[str, tuple[TouchpointCase, ...]]:
    source = cases if cases is not None else ALL_TOUCHPOINT_CASES
    grouped: dict[str, list[TouchpointCase]] = {}
    for case in source:
        grouped.setdefault(case.degradation, []).append(case)
    return {degradation: tuple(items) for degradation, items in grouped.items()}


__all__ = [
    "ALL_TOUCHPOINT_CASES",
    "DECIDE_GATE_CASES",
    "PLANNER_CASES",
    "REVIEWER_CASES",
    "ROUTE_INSTRUCTION_CASES",
    "TouchpointCase",
    "cases_by_degradation",
    "cases_by_touchpoint",
]
