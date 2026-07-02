"""Regression gate for the TST-1 degraded-output eval corpus.

Drives every marvis.orchestrator.eval.touchpoint_cases.TouchpointCase
through the *real* production function it names (decide_gate,
route_instruction, Planner.generate, Reviewer.llm_critique) -- never a
reimplementation -- and asserts the documented outcome.

Cases without ``expected_failure`` lock in a currently-safe degraded-input
path: if a future change to json_reply.py / auto_drive.py /
instruction_router.py / planner.py / reviewer.py silently breaks the safe
fallback, this test catches it.

Cases *with* ``expected_failure`` document a currently-real, confirmed-unsafe
behavior. They assert the unsafe outcome on purpose, so this file doubles as
an executable "known gaps" ledger: if someone fixes one of these paths, this
test starts failing (the case's asserted outcome no longer happens) and the
fix author must consciously update/remove the expected_failure marker rather
than a fix silently landing unnoticed.
"""

from __future__ import annotations

import pytest

from marvis.agent.auto_drive import decide_gate
from marvis.agent.instruction_router import route_instruction
from marvis.orchestrator.capability import resolve_tier
from marvis.orchestrator.contracts import PlanStep
from marvis.orchestrator.eval.runner import build_tool_registry
from marvis.orchestrator.eval.touchpoint_cases import (
    ALL_TOUCHPOINT_CASES,
    DECIDE_GATE_CASES,
    PLANNER_CASES,
    REVIEWER_CASES,
    ROUTE_INSTRUCTION_CASES,
    cases_by_degradation,
    cases_by_touchpoint,
)
from marvis.orchestrator.planner import Planner, PlanningError
from marvis.orchestrator.reviewer import Reviewer
from marvis.orchestrator.validator import PlanValidator
from marvis.plugins.manifest import ToolRef


class _ScriptedLLM:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.payload


def _assert_subset(actual: dict, expected: dict, *, case_id: str) -> None:
    for key, value in expected.items():
        assert key in actual, f"{case_id}: missing key {key!r} in {actual!r}"
        assert actual[key] == value, (
            f"{case_id}: {key}={actual[key]!r} does not match expected {value!r}"
        )


# -- matrix coverage: touchpoint x degradation -------------------------------

def test_touchpoint_matrix_covers_all_four_touchpoints_with_six_degradations_each():
    by_touchpoint = cases_by_touchpoint()
    assert set(by_touchpoint) == {"decide_gate", "route_instruction", "planner", "reviewer"}
    for touchpoint, cases in by_touchpoint.items():
        degradations = {case.degradation for case in cases}
        assert len(degradations) >= 6, f"{touchpoint} only covers {sorted(degradations)}"

    by_degradation = cases_by_degradation()
    expected_degradations = {
        "markdown_fence",
        "prose_prefix_suffix",
        "key_casing",
        "truncated_json",
        "negation_semantics",
        "think_tag_contamination",
    }
    assert expected_degradations.issubset(by_degradation)


def test_touchpoint_case_ids_are_unique():
    ids = [case.id for case in ALL_TOUCHPOINT_CASES]
    assert len(ids) == len(set(ids))


# -- decide_gate --------------------------------------------------------------

@pytest.mark.parametrize("case", DECIDE_GATE_CASES, ids=lambda case: case.id)
def test_decide_gate_touchpoint_cases(case):
    llm = _ScriptedLLM(case.raw_output)
    gate = {"content": "特征筛选完成", "metadata": {}}

    decision = decide_gate(llm, gate=gate)

    _assert_subset(decision, case.expected, case_id=case.id)


# -- route_instruction ----------------------------------------------------

@pytest.mark.parametrize("case", ROUTE_INSTRUCTION_CASES, ids=lambda case: case.id)
def test_route_instruction_touchpoint_cases(case):
    llm = _ScriptedLLM(case.raw_output)

    result = route_instruction(llm, gate_context="调参节点", instruction="用户指令")

    _assert_subset(result, case.expected, case_id=case.id)


# -- planner ------------------------------------------------------------------

@pytest.fixture(scope="module")
def _planner_deps():
    registry = build_tool_registry()
    validator = PlanValidator(registry)
    return registry, validator


@pytest.mark.parametrize("case", PLANNER_CASES, ids=lambda case: case.id)
def test_planner_generate_touchpoint_cases(case, _planner_deps):
    _registry, validator = _planner_deps
    registry = _registry
    llm = _ScriptedLLM(case.raw_output)
    planner = Planner(registry, lambda: llm, validator)
    tier = resolve_tier("balanced")

    if case.expected.get("raises") == "PlanningError":
        with pytest.raises(PlanningError):
            planner.generate("goal", task_id="t", memory_context={}, task_context={}, tier=tier)
        assert len(llm.calls) == case.expected["call_count"], (
            f"{case.id}: expected {case.expected['call_count']} attempts, got {len(llm.calls)}"
        )
    else:
        plan = planner.generate("goal", task_id="t", memory_context={}, task_context={}, tier=tier)
        assert plan is not None
        assert len(llm.calls) == case.expected["call_count"]


# -- reviewer -------------------------------------------------------------

def _reviewer_step() -> PlanStep:
    return PlanStep(
        id="s1",
        plan_id="p1",
        index=0,
        title="train",
        tool_ref=ToolRef("modeling", "train_model"),
        inputs={},
        depends_on=[],
        post_checks=[],
    )


@pytest.mark.parametrize("case", REVIEWER_CASES, ids=lambda case: case.id)
def test_reviewer_llm_critique_touchpoint_cases(case):
    llm = _ScriptedLLM(case.raw_output)

    verdict = Reviewer(lambda: llm).llm_critique(_reviewer_step(), {"ks": 0.1}, "goal")

    actual = {"passed": verdict.passed, "reasons": verdict.reasons}
    _assert_subset(actual, case.expected, case_id=case.id)


# -- expected_failure bookkeeping --------------------------------------------

def test_expected_failure_cases_carry_a_non_empty_explanation():
    for case in ALL_TOUCHPOINT_CASES:
        if case.expected_failure:
            assert len(case.expected_failure) > 20, case.id


def test_expected_failure_count_matches_reported_unsafe_touchpoints():
    # Locks the count so silently adding/removing a documented gap is a
    # visible diff, not something that drifts unnoticed.
    unsafe = [case.id for case in ALL_TOUCHPOINT_CASES if case.expected_failure]
    # 11 documented at authoring time; the three think-tag cases flipped to safe
    # once LLM-6's client-side <think> stripping landed.
    assert len(unsafe) == 8, unsafe
