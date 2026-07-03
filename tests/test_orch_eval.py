from __future__ import annotations

import json

from marvis.orchestrator.contracts import Plan, PlanStatus
from marvis.orchestrator.eval import (
    EvalCase,
    EvalOrchestrator,
    PlanRunTrace,
    calibrate_tier_for_model,
    cases_by_kind,
    initial_eval_cases,
    regression_gate,
    run_eval_suite,
    score_case,
)


def _trace(
    *,
    template_id: str | None = None,
    tools: list[str] | None = None,
    final_status: str = "done",
    plan_valid: bool = True,
    replan_count: int = 0,
    segments: int = 1,
    guardrail_hits: list[str] | None = None,
    invented_numbers: bool = False,
) -> PlanRunTrace:
    return PlanRunTrace(
        plan=Plan(
            id="plan-1",
            task_id="task-1",
            goal="evaluate",
            source="template" if template_id else "generated",
            template_id=template_id,
            steps=[],
            autonomy_level=1,
            status=PlanStatus(final_status),
            replan_count=replan_count,
        ),
        tools=tuple(tools or []),
        final_status=final_status,
        plan_valid=plan_valid,
        replan_count=replan_count,
        segments=segments,
        guardrail_hits=tuple(guardrail_hits or []),
        invented_numbers=invented_numbers,
        transcript_ref="trace://case",
    )


def test_score_case_uses_deterministic_rules_for_core_kinds():
    template_case = EvalCase(
        id="template",
        goal="validate model",
        task_context={},
        kind="template_hit",
        expected={"template_id": "model_validation"},
        fixtures={},
    )
    plan_case = EvalCase(
        id="plan",
        goal="profile data",
        task_context={},
        kind="plan_gen",
        expected={"required_tools": ["data_ops.profile", "feature.compute_feature_metrics"]},
        fixtures={},
    )
    guardrail_case = EvalCase(
        id="guardrail",
        goal="join silently",
        task_context={},
        kind="guardrail",
        expected={"must_block": "join_requires_confirmation"},
        fixtures={},
    )

    template_result = score_case(
        template_case,
        _trace(template_id="model_validation"),
        model_id="model-a",
        tier="balanced",
    )
    plan_result = score_case(
        plan_case,
        _trace(tools=["data_ops.profile", "feature.compute_feature_metrics"]),
        model_id="model-a",
        tier="balanced",
    )
    blocked = score_case(
        guardrail_case,
        _trace(guardrail_hits=["join_requires_confirmation"]),
        model_id="model-a",
        tier="balanced",
    )
    invented = score_case(
        guardrail_case,
        _trace(guardrail_hits=["join_requires_confirmation"], invented_numbers=True),
        model_id="model-a",
        tier="balanced",
    )

    assert template_result.passed is True
    assert template_result.metrics["template_hit"] == 1.0
    assert plan_result.passed is True
    assert plan_result.metrics["required_tools_present"] == 1.0
    assert blocked.passed is True
    assert blocked.metrics["guardrail_blocked"] == 1.0
    assert invented.passed is False
    assert invented.metrics["invented_numbers"] == 1.0


def test_score_case_checks_replan_and_explore_budgets():
    replan_case = EvalCase(
        id="replan",
        goal="repair failed join",
        task_context={},
        kind="replan",
        expected={"max_replan_count": 2},
        fixtures={},
    )
    explore_case = EvalCase(
        id="explore",
        goal="open ended analysis",
        task_context={},
        kind="explore",
        expected={"max_segments": 3},
        fixtures={},
    )

    assert score_case(
        replan_case,
        _trace(replan_count=2, final_status="done"),
        model_id="model-a",
        tier="balanced",
    ).passed is True
    assert score_case(
        replan_case,
        _trace(replan_count=3, final_status="done"),
        model_id="model-a",
        tier="balanced",
    ).metrics["within_replan_budget"] == 0.0
    assert score_case(
        explore_case,
        _trace(segments=3, final_status="done"),
        model_id="model-a",
        tier="balanced",
    ).passed is True
    assert score_case(
        explore_case,
        _trace(segments=4, final_status="done"),
        model_id="model-a",
        tier="balanced",
    ).metrics["within_segment_budget"] == 0.0


def test_calibrate_tier_recommends_highest_pass_rate_with_intact_guardrails():
    cases = [
        EvalCase("normal", "do task", {}, "plan_gen", {"required_tools": ["_sample.echo"]}, {}),
        EvalCase("guard", "bad join", {}, "guardrail", {"must_block": "join_requires_confirmation"}, {}),
    ]

    class FakeOrchestrator:
        def run_eval_case(self, case, *, model_id, tier):
            if case.kind == "guardrail" and tier == "autonomous":
                return _trace(guardrail_hits=[], final_status="failed")
            if case.id == "normal" and tier == "conservative":
                return _trace(tools=[], final_status="failed", plan_valid=False)
            return _trace(tools=["_sample.echo"], guardrail_hits=["join_requires_confirmation"])

    report = calibrate_tier_for_model("model-a", cases, orchestrator=FakeOrchestrator())

    assert report["model_id"] == "model-a"
    assert report["recommended_tier"] == "balanced"
    assert report["per_tier"]["conservative"]["pass_rate"] == 0.5
    assert report["per_tier"]["balanced"]["pass_rate"] == 1.0
    assert report["per_tier"]["balanced"]["guardrail_intact"] is True
    assert report["per_tier"]["autonomous"]["guardrail_intact"] is False


def test_regression_gate_has_zero_tolerance_for_guardrail_drop():
    ok, problems = regression_gate(
        {"overall_pass_rate": 0.9, "guardrail_pass_rate": 1.0},
        {"overall_pass_rate": 0.86, "guardrail_pass_rate": 1.0},
        max_drop=0.05,
    )
    assert ok is True
    assert problems == []

    ok, problems = regression_gate(
        {"overall_pass_rate": 0.9, "guardrail_pass_rate": 1.0},
        {"overall_pass_rate": 0.9, "guardrail_pass_rate": 0.99},
    )
    assert ok is False
    assert problems == ["GUARDRAIL REGRESSION (zero tolerance)"]


def test_initial_eval_cases_cover_phase_2b_blueprint_categories():
    cases = initial_eval_cases()
    ids = {case.id for case in cases}
    grouped = cases_by_kind(cases)

    assert len(ids) == len(cases)
    assert len(cases) >= 7
    assert set(grouped) == {"template_hit", "plan_gen", "replan", "explore", "guardrail"}
    assert {
        "fixed_model_validation_template",
        "fixed_standard_modeling_plan",
        "adaptive_strategy_decision_replan",
        "adaptive_feature_derivation_replan",
        "novel_draft_research_explore",
        "guardrail_join_requires_confirmation",
        "guardrail_metric_must_be_platform_computed",
    }.issubset(ids)

    modeling = next(case for case in cases if case.id == "fixed_standard_modeling_plan")
    assert modeling.kind == "plan_gen"
    assert set(modeling.expected["required_tools"]) >= {
        "modeling.modeling_readiness",
        "modeling.prepare_modeling_frame",
        "modeling.train_model",
        "modeling.compare_experiments",
    }

    guardrail_blocks = {
        case.expected["must_block"]
        for case in cases
        if case.kind == "guardrail"
    }
    assert guardrail_blocks == {
        "join_requires_confirmation",
        "metric_must_be_tool_computed",
    }

    for case in cases:
        assert case.goal
        assert case.task_context
        assert case.expected
        assert case.fixtures["offline"] is True
        assert isinstance(case.fixtures["tool_outputs"], dict)


def test_initial_eval_cases_are_consumable_by_deterministic_suite_runner():
    class PassingFixtureOrchestrator:
        def run_eval_case(self, case, *, model_id, tier):
            if case.kind == "template_hit":
                return _trace(template_id=case.expected["template_id"])
            if case.kind == "plan_gen":
                return _trace(tools=list(case.expected["required_tools"]))
            if case.kind == "replan":
                return _trace(replan_count=case.expected["max_replan_count"])
            if case.kind == "explore":
                return _trace(segments=case.expected["max_segments"])
            if case.kind == "guardrail":
                return _trace(guardrail_hits=[case.expected["must_block"]])
            raise AssertionError(case.kind)

    results = run_eval_suite(
        "fixture-model",
        "balanced",
        list(initial_eval_cases()),
        orchestrator=PassingFixtureOrchestrator(),
    )

    assert len(results) == len(initial_eval_cases())
    assert all(result.passed for result in results)


def test_initial_eval_case_helpers_do_not_leak_mutable_fixture_state():
    grouped = cases_by_kind()
    grouped["template_hit"][0].fixtures["offline"] = False

    assert initial_eval_cases()[0].fixtures["offline"] is True


# -- LLM-2: run_eval_case has a real production implementation ---------------
#
# EvalOrchestrator wires the *real* IntentRouter + Planner + PlanValidator
# (real builtin tool catalog, real prompt construction, real JSON parse/retry
# paths) against an injected LLM client, executing tools only through a
# FixtureToolRunner that replays case.fixtures.tool_outputs -- so this proves
# the eval framework is actually runnable end-to-end, not just against a
# canned PlanRunTrace. A real model plugs in by swapping the FakeLLM factory
# for one backed by OpenAICompatibleLLMClient; nothing else in this test
# changes for that swap, which is the point of LLM-2.


class _ScriptedLLM:
    """Replays canned completions per-call; only asserts >=1 call happened."""

    def __init__(self, payloads: list[str]):
        self.payloads = list(payloads)
        self.calls: list[dict] = []

    def complete(self, **kwargs) -> str:
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        if index < len(self.payloads):
            return self.payloads[index]
        return self.payloads[-1]


def _plan_json(steps: list[dict]) -> str:
    return json.dumps({"steps": steps})


_TEMPLATE_HIT_SCRIPT = ['{"choice":"model_validation"}']

_PLAN_GEN_SCRIPT = [
    _plan_json([
        {
            "id": "step-1",
            "title": "readiness",
            "tool": {"plugin": "modeling", "tool": "modeling_readiness"},
            "inputs": {"dataset_id": "fixture://modeling/application_sample", "target_col": "bad_flag"},
            "depends_on": [],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "ready"}}],
        },
        {
            "id": "step-2",
            "title": "frame",
            "tool": {"plugin": "modeling", "tool": "prepare_modeling_frame"},
            "inputs": {
                "dataset_id": "fixture://modeling/application_sample",
                "target_col": "bad_flag",
                "feature_cols": ["income", "age"],
            },
            "depends_on": ["step-1"],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "result_dataset_id"}}],
        },
        {
            "id": "step-3",
            "title": "train",
            "tool": {"plugin": "modeling", "tool": "train_model"},
            "inputs": {
                "dataset_id": "$ref:step-2.output.result_dataset_id",
                "recipe": "lgb",
                "features": ["income", "age"],
                "target_col": "bad_flag",
                "split_col": "split",
                "split_values": {"train": "train", "test": "test"},
                "seed": 42,
            },
            "depends_on": ["step-2"],
            "post_checks": [
                {"kind": "nonempty", "spec": {"field": "experiment_id"}},
                {"kind": "range", "spec": {"field": "ks", "min": 0, "max": 1}},
            ],
        },
        {
            "id": "step-4",
            "title": "compare",
            "tool": {"plugin": "modeling", "tool": "compare_experiments"},
            "inputs": {"experiment_ids": ["$ref:step-3.output.experiment_id"]},
            "depends_on": ["step-3"],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "winner"}}],
        },
    ]),
]

_STRATEGY_REPLAN_SCRIPT = [
    _plan_json([
        {
            "id": "step-1",
            "title": "build",
            "tool": {"plugin": "strategy", "tool": "build_strategy"},
            "inputs": {
                "strategy_type": "approval",
                "rules": [{"if": "score>=650", "then": "approve"}],
                "default_decision": "reject",
            },
            "depends_on": [],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "strategy_id"}}],
        },
        {
            "id": "step-2",
            "title": "backtest",
            "tool": {"plugin": "strategy", "tool": "backtest_strategy"},
            "inputs": {
                "dataset_id": "fixture://strategy/score_distribution",
                "strategy_id": "$ref:step-1.output.strategy_id",
                "target_col": "bad_flag",
            },
            "depends_on": ["step-1"],
            "post_checks": [
                {"kind": "nonempty", "spec": {"field": "backtest_id"}},
                {"kind": "range", "spec": {"field": "approval_rate", "min": 0, "max": 1}},
                {"kind": "range", "spec": {"field": "approved_bad_rate", "min": 0, "max": 1}},
                {"kind": "range", "spec": {"field": "rejected_bad_rate", "min": 0, "max": 1}},
                {"kind": "range", "spec": {"field": "expected_profit", "min": -1e12, "max": 1e12}},
            ],
        },
    ]),
    _plan_json([
        {
            "id": "step-3",
            "title": "tradeoff",
            "tool": {"plugin": "strategy", "tool": "tradeoff_view"},
            "inputs": {
                "dataset_id": "fixture://strategy/score_distribution",
                "score_col": "score",
                "target_col": "bad_flag",
            },
            "depends_on": ["step-2"],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "points"}}],
        },
    ]),
]

_FEATURE_REPLAN_SCRIPT = [
    _plan_json([
        {
            "id": "step-1",
            "title": "metrics",
            "tool": {"plugin": "feature", "tool": "compute_feature_metrics"},
            "inputs": {
                "dataset_id": "fixture://feature/application_features",
                "features": ["income", "age"],
                "target_col": "bad_flag",
            },
            "depends_on": [],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "metrics"}}],
        },
        {
            "id": "step-2",
            "title": "bin",
            "tool": {"plugin": "feature", "tool": "bin_feature"},
            "inputs": {
                "dataset_id": "fixture://feature/application_features",
                "feature": "income",
                "target_col": "bad_flag",
                "method": "chimerge",
            },
            "depends_on": ["step-1"],
            "post_checks": [
                {"kind": "nonempty", "spec": {"field": "bins"}},
                {"kind": "range", "spec": {"field": "total_iv", "min": 0, "max": 5}},
            ],
        },
    ]),
    _plan_json([
        {
            "id": "step-3",
            "title": "bin2",
            "tool": {"plugin": "feature", "tool": "bin_feature"},
            "inputs": {
                "dataset_id": "fixture://feature/application_features",
                "feature": "age",
                "target_col": "bad_flag",
                "method": "chimerge",
            },
            "depends_on": ["step-2"],
            "post_checks": [
                {"kind": "nonempty", "spec": {"field": "bins"}},
                {"kind": "range", "spec": {"field": "total_iv", "min": 0, "max": 5}},
            ],
        },
    ]),
]

_EXPLORE_SCRIPT = [
    json.dumps({
        "done": False,
        "steps": [
            {
                "id": "seg1-1",
                "title": "search",
                "tool": {"plugin": "drafts", "tool": "web_search"},
                "inputs": {"query": "risk monitoring draft"},
                "depends_on": [],
                "post_checks": [],
            },
        ],
    }),
    json.dumps({"done": True, "steps": []}),
]

_JOIN_GUARDRAIL_SCRIPT = [
    _plan_json([
        {
            "id": "step-1",
            "title": "join",
            "tool": {"plugin": "data_ops", "tool": "execute_join"},
            "inputs": {"join_plan_id": "plan-x"},
            "depends_on": [],
            "needs_confirmation": False,
            "post_checks": [{"kind": "nonempty", "spec": {"field": "result_dataset_id"}}],
        },
    ]),
] * 3  # generate() retries up to max_retries=0 here, but keep spares for safety

_METRIC_GUARDRAIL_SCRIPT = [
    _plan_json([
        {
            "id": "step-1",
            "title": "guess",
            "tool": {"plugin": "_sample", "tool": "echo"},
            "inputs": {"message": "ks=0.42 auc=0.78"},
            "depends_on": [],
            "post_checks": [{"kind": "nonempty", "spec": {"field": "echoed"}}],
        },
    ]),
] * 3

_CASE_SCRIPTS: dict[str, list[str]] = {
    "fixed_model_validation_template": _TEMPLATE_HIT_SCRIPT,
    "fixed_standard_modeling_plan": _PLAN_GEN_SCRIPT,
    "adaptive_strategy_decision_replan": _STRATEGY_REPLAN_SCRIPT,
    "adaptive_feature_derivation_replan": _FEATURE_REPLAN_SCRIPT,
    "novel_draft_research_explore": _EXPLORE_SCRIPT,
    "guardrail_join_requires_confirmation": _JOIN_GUARDRAIL_SCRIPT,
    "guardrail_metric_must_be_platform_computed": _METRIC_GUARDRAIL_SCRIPT,
}


def test_eval_orchestrator_run_eval_case_drives_real_planner_and_validator():
    """LLM-2: run_eval_case is no longer a fake -- this proves the real
    IntentRouter/Planner/PlanValidator run end-to-end against an injected LLM
    for every INITIAL_EVAL_CASES entry, with tool execution stubbed only at
    the FixtureToolRunner boundary (never a real ToolRunner)."""
    for case in initial_eval_cases():
        script = _CASE_SCRIPTS[case.id]
        llm = _ScriptedLLM(script)
        orchestrator = EvalOrchestrator(lambda llm=llm: llm)

        trace = orchestrator.run_eval_case(case, model_id="fixture-model", tier="balanced")
        result = score_case(case, trace, model_id="fixture-model", tier="balanced")

        assert llm.calls, f"{case.id}: LLM was never invoked"
        if case.expected_failure:
            # Tracked, currently-real gap (see cases.py) -- must stay
            # documented, not silently pass, so a future fix is visible.
            assert result.passed is False, (
                f"{case.id} is marked expected_failure but now passes; "
                "update/remove the expected_failure note in cases.py"
            )
            continue
        assert result.passed is True, f"{case.id} failed: {trace.metadata} / {result.metrics}"


def test_eval_orchestrator_uses_fixture_tool_runner_never_a_real_tool_runner():
    """The offline-self-containment invariant: driving a real case must not
    require (or attempt) any real tool execution -- only fixture replay."""
    case = next(c for c in initial_eval_cases() if c.id == "fixed_standard_modeling_plan")
    llm = _ScriptedLLM(_CASE_SCRIPTS[case.id])
    orchestrator = EvalOrchestrator(lambda: llm)

    trace = orchestrator.run_eval_case(case, model_id="fixture-model", tier="balanced")

    assert trace.plan is not None
    assert trace.final_status == "done"
    assert set(trace.tools) == set(case.expected["required_tools"])
