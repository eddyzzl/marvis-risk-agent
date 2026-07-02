"""Degraded-output eval corpus for the weak-model JSON touchpoints (TST-1).

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
        expected={"action": "halt", "reason": "无法解析模型决策,转人工确认。"},
        notes="Unparseable -> safe halt fallback; decide_gate's retry gets a fresh chance.",
    ),
    TouchpointCase(
        id="decide_gate_negation_in_reason_contradicts_action",
        touchpoint="decide_gate",
        degradation="negation_semantics",
        raw_output=(
            '{"action":"confirm","reason":"指标异常，不能确认，不可以继续。"}'
        ),
        expected={"action": "confirm"},
        expected_failure=(
            "_parse_decision trusts the structured action field literally and "
            "never cross-checks it against the natural-language reason. A "
            "model that writes action=confirm but says (in Chinese) 'metrics "
            "abnormal, cannot confirm, must not continue' in its own reason "
            "field is auto-confirmed -- the negation is silently ignored. "
            "This is distinct from the deterministic string-guard in "
            "marvis/agent/service.py / plan_driver.py, which covers a "
            "different flow (manual free-text replies), not decide_gate."
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
        expected={"action": "confirm"},
        expected_failure=(
            "_parse_route trusts the action field literally with no "
            "cross-check against reason. A model whose reason literally says "
            "the user disagrees and instructed not to continue, but whose "
            "action field still says confirm, is routed as confirm."
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
# marvis/orchestrator/planner.py's _parse_plan_json / _parse_steps_json /
# _parse_json_object all call bare json.loads(raw) -- NOT
# marvis.agent.json_reply.load_json_object. There is no fence-stripping, no
# <think> handling, and no lenient-candidate extraction anywhere in the
# planner. The only tolerance is the retry loop (max_retries=2 for
# generate(), MAX_REPLAN_PARSE_RETRY=1 for replan()/next_explore_segment())
# which re-prompts with last_error text but resends the exact same
# generation request, so a model that systematically wraps its output in a
# fence will fail every attempt and raise PlanningError (a fatal error a
# caller must catch), not degrade to an empty/safe plan.

PLANNER_CASES: tuple[TouchpointCase, ...] = (
    TouchpointCase(
        id="planner_fence_exhausts_retries",
        touchpoint="planner",
        degradation="markdown_fence",
        raw_output='```json\n{"steps": []}\n```',
        expected={"raises": "PlanningError", "call_count": 3},
        expected_failure=(
            "_parse_plan_json uses bare json.loads(raw), not "
            "marvis.agent.json_reply.load_json_object -- there is zero fence "
            "tolerance anywhere in the planner. A model that always fences "
            "its JSON (a very common weak-model / serving-template default) "
            "fails json.loads on every one of max_retries+1=3 attempts and "
            "generate() raises PlanningError, aborting plan generation "
            "entirely instead of degrading to a safe empty/halted state. The "
            "retry loop's last_error feedback ('not json: Expecting value...') "
            "gives the model no actionable signal to strip the fence, since "
            "it already believes it answered in valid JSON."
        ),
    ),
    TouchpointCase(
        id="planner_prose_prefix_suffix_exhausts_retries",
        touchpoint="planner",
        degradation="prose_prefix_suffix",
        raw_output='计划如下：\n{"steps": []}\n以上。',
        expected={"raises": "PlanningError", "call_count": 3},
        expected_failure=(
            "Same bare-json.loads gap as the fence case: any chat-style "
            "preamble/postscript around the JSON object makes every retry "
            "attempt fail identically."
        ),
    ),
    TouchpointCase(
        id="planner_truncated_json_exhausts_retries",
        touchpoint="planner",
        degradation="truncated_json",
        raw_output='{"steps": [{"title": "a"',
        expected={"raises": "PlanningError", "call_count": 3},
        notes=(
            "Also unsafe in the same way as fence/prefix, but this one is "
            "arguably reasonable to leave a hard failure (a truncated plan "
            "cannot be completed by any retry strategy that resends the same "
            "prompt); recorded for matrix completeness, not flagged as a "
            "distinct new risk beyond the fence/prefix findings."
        ),
        expected_failure=(
            "Bare json.loads on a truncated payload predictably raises "
            "PlanningError on every retry attempt; no partial-plan recovery "
            "exists."
        ),
    ),
    TouchpointCase(
        id="planner_think_tag_contaminates_every_retry",
        touchpoint="planner",
        degradation="think_tag_contamination",
        raw_output=(
            "<think>这个任务需要先读取数据"
            "</think>"
            '{"steps": []}'
        ),
        expected={"raises": "PlanningError", "call_count": 3},
        expected_failure=(
            "A <think>...</think> prefix before the JSON object is not valid "
            "JSON on its own (json.loads sees the whole string, including "
            "the <think> text, as the payload) -- bare json.loads fails "
            "identically to the fence case on every retry attempt."
        ),
    ),
    TouchpointCase(
        id="planner_key_casing_is_schema_error_not_parse_error",
        touchpoint="planner",
        degradation="key_casing",
        raw_output='{"Steps": []}',
        expected={"raises": "PlanningError", "call_count": 3},
        notes=(
            "This one parses as JSON (json.loads succeeds) but "
            "_parse_plan_json only reads data.get('steps') (lowercase); "
            "'Steps' is a different key so raw_steps is None -> "
            "PlanningError('plan JSON must include non-empty steps'). Same "
            "end state as the other planner cases but for a different "
            "reason (schema mismatch, not a JSON syntax error) -- recorded "
            "for matrix completeness rather than as an additional distinct "
            "risk."
        ),
    ),
    TouchpointCase(
        id="planner_negation_semantics_not_applicable",
        touchpoint="planner",
        degradation="negation_semantics",
        raw_output='{"steps": []}',
        expected={"raises": "PlanningError", "call_count": 3},
        notes=(
            "The planner has no confirm/halt action field -- negation "
            "semantics is a decide_gate/route_instruction concept, not a "
            "plan-shape concept. An empty-steps plan is syntactically valid "
            "JSON (parses fine) but fails PlanValidator/_parse_plan_json's "
            "own non-empty-steps requirement, which is an orthogonal, "
            "already-enforced guard -- included here only so the touchpoint "
            "matrix has an explicit entry for this category rather than a "
            "silent gap."
        ),
        expected_failure=(
            "An empty-steps plan JSON fails the non-empty check on every "
            "retry and raises PlanningError, same as the other planner "
            "degradations; listed as not-applicable to negation specifically "
            "but still surfaces the identical bare-json.loads/no-partial-"
            "recovery behavior."
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
        expected={"passed": True, "reasons": []},
        notes=(
            "data.get('passed', True) DEFAULTS TO TRUE when the (wrong-case) "
            "key is absent -- unlike decide_gate/route_instruction (which "
            "default an unrecognized action to a safe halt/clarify), a "
            "wrong-case reviewer reply defaults to passed=True. This is "
            "lower-severity than the other findings because AGT-3 already "
            "bounds llm_critique/final_review to never single-handedly FAIL "
            "a plan, but it is worth flagging: the reviewer's own polarity "
            "default is 'optimistic', opposite of the other three "
            "touchpoints' 'pessimistic' defaults."
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
        expected={"passed": True},
        expected_failure=(
            "_parse_soft_verdict reads data.get('passed', True) literally "
            "with no cross-check against the reasons text. A model whose "
            "reasons list says (in Chinese) 'metrics abnormal, should not "
            "pass, recommend retraining' but whose passed field still says "
            "true is reported as passed=True. Bounded impact only by AGT-3 "
            "(an LLM verdict cannot unilaterally fail a plan), not by any "
            "textual consistency check in the reviewer itself."
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
