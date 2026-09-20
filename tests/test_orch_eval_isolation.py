from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from marvis.orchestrator.contracts import PlanStatus, StepStatus
from marvis.orchestrator.eval.contracts import EvalCase, PlanRunTrace
from marvis.orchestrator.eval.cases import initial_eval_cases
from marvis.orchestrator.eval import scoring as eval_scoring
from marvis.orchestrator.eval.runner import EvalOrchestrator, build_tool_registry
from marvis.orchestrator.eval.scoring import calibrate_tier_for_model, regression_gate, score_case


class RecordingLLM:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def complete(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    return build_tool_registry(db_path=tmp_path_factory.mktemp("eval-isolation") / "eval.sqlite")


def echo(step_id="step-1", *, message="business input", depends_on=()):
    return {
        "id": step_id,
        "title": "Echo approved input",
        "tool": {"plugin": "_sample", "tool": "echo"},
        "inputs": {"message": message},
        "depends_on": list(depends_on),
        "post_checks": [],
    }


def case_for(kind):
    return EvalCase(
        id="isolation",
        goal="Echo the approved business input, then finish.",
        task_context={
            "workflow_family": "novel",
            "decision_point_after": "_sample.echo",
            "historical_context": "Historical context only. " * 1800,
        },
        kind=kind,
        expected={"required_tools": ["_sample.echo"], "max_segments": 2},
        fixtures={"tool_outputs": {"_sample.echo": {"echoed": "business input"}}},
    )


@pytest.mark.parametrize("kind", ["template_hit", "plan_gen", "replan", "explore", "guardrail"])
def test_expected_never_changes_model_requests_in_blind_runs(kind, catalog):
    """Compare full requests, including retries/compression/replan ledgers."""
    case = case_for(kind)
    if kind == "template_hit":
        case = initial_eval_cases()[0]
        replies = ['{"choice":"model_validation"}']
    elif kind == "explore":
        replies = ["invalid JSON", json.dumps({"done": False, "steps": [echo()]}), '{"done":true,"steps":[]}']
    elif kind == "replan":
        replies = ["invalid JSON", json.dumps({"steps": [echo()]}), "invalid JSON", json.dumps({"steps": [echo("step-2", depends_on=["step-1"])]})]
    else:
        replies = ["invalid JSON", json.dumps({"steps": [echo()]})]
    sentinel = "SECRET_EXPECTED_ONLY_3d86a9"
    changed = replace(case, expected={
        "required_tools": ["_sample.echo"],
        "required_tool_inputs": [{"tool": "_sample.echo", "inputs": {"message": sentinel}}],
        "forbidden_tools": ["_sample.fail"],
        "max_segments": 0,
        "must_block": sentinel,
    })
    recorded = []
    traces = []
    for variant in (case, changed):
        llm = RecordingLLM(replies)
        orchestrator = EvalOrchestrator(lambda: llm, tool_registry=catalog, model_source="fixture_model")
        traces.append(orchestrator.run_eval_case(variant, model_id="fixture", tier="balanced"))
        recorded.append(llm.calls)
    assert recorded[0]
    assert recorded[0] == recorded[1]
    assert sentinel not in json.dumps(recorded)
    assert traces[0].evaluation_mode == "blind"
    if kind in {"replan", "explore"}:
        assert traces[0].final_status == traces[1].final_status == "simulated_done"
        assert len(recorded[0]) >= 3


def test_invalid_expected_is_checked_only_after_blind_model_run(catalog):
    case = case_for("plan_gen")
    replies = [json.dumps({"steps": [echo()]})]
    requests = []
    for expected in (case.expected, {"required_tools": ["missing.answer_sentinel"]}):
        llm = RecordingLLM(replies)
        trace = EvalOrchestrator(lambda: llm, tool_registry=catalog).run_eval_case(
            replace(case, expected=expected), model_id="fixture", tier="balanced"
        )
        requests.append(llm.calls)
    assert requests[0] == requests[1]
    assert trace.final_status == "harness_error"
    assert trace.metadata["failure_stage"] == "catalog_postflight"


@pytest.mark.parametrize("kind", ["plan_gen", "replan", "explore"])
@pytest.mark.parametrize("outputs", [{}, {"_sample.echo": {}}, {"_sample.echo": None}])
def test_missing_fixture_output_fails_without_claiming_execution(kind, outputs, catalog):
    case = replace(case_for(kind), fixtures={"tool_outputs": outputs})
    llm = RecordingLLM([json.dumps({"done": False, "steps": [echo()]})])
    trace = EvalOrchestrator(lambda: llm, tool_registry=catalog).run_eval_case(
        case, model_id="fixture", tier="balanced"
    )
    assert trace.final_status == "simulation_unmodeled"
    assert trace.plan.status == PlanStatus.FAILED
    assert all(step.status != StepStatus.DONE for step in trace.plan.steps)
    assert trace.execution_mode == "fixture_simulation"
    assert trace.executor_invoked is False
    result = score_case(case, trace)
    assert result.passed is False
    assert result.metrics["real_execution_completed"] == 0
    assert result.metrics["simulation_completed"] == 0


def test_fixture_success_and_legacy_done_never_count_as_real_execution(catalog):
    case = case_for("plan_gen")
    llm = RecordingLLM([json.dumps({"steps": [echo()]})])
    trace = EvalOrchestrator(lambda: llm, tool_registry=catalog).run_eval_case(
        case, model_id="fixture", tier="balanced"
    )
    assert trace.final_status == "simulated_done"
    assert trace.plan.status != PlanStatus.DONE
    assert score_case(case, trace).metrics["simulation_completed"] == 1
    legacy = PlanRunTrace(plan=trace.plan, tools=trace.tools, final_status="done", plan_valid=True)
    for candidate in (trace, legacy):
        result = score_case(case, candidate)
        assert result.metrics["real_execution_completed"] == 0
        assert result.executor_invoked is False
    assert score_case(case, legacy).execution_mode == "unknown"


def test_unmodeled_fixture_stays_in_failure_denominator(catalog):
    case = replace(case_for("plan_gen"), fixtures={"tool_outputs": {}})
    llm = RecordingLLM([json.dumps({"steps": [echo()]})])
    orchestrator = EvalOrchestrator(
        lambda: llm, tool_registry=catalog, model_source="fixture_model"
    )
    report = calibrate_tier_for_model("fixture", [case], orchestrator=orchestrator)
    assert report["status"] == "INCOMPLETE"
    assert report["simulation_unmodeled_count"] == 3
    assert report["recommended_tier"] is None
    for tier in report["per_tier"].values():
        assert tier["scored_case_count"] == 1
        assert tier["harness_excluded_case_count"] == 0
        assert tier["pass_rate"] == 0


def test_old_custom_runner_cannot_receive_an_inferred_capability_recommendation():
    class LegacyRunner:
        def run_eval_case(self, case, **_kwargs):
            return PlanRunTrace(
                plan=None, final_status="blocked", guardrail_hits=("guard",)
            )

    case = replace(case_for("guardrail"), expected={"must_block": "guard"})
    report = calibrate_tier_for_model("legacy", [case], orchestrator=LegacyRunner())
    assert all(tier["pass_rate"] == 1 for tier in report["per_tier"].values())
    assert report["evaluation_mode"] == "unknown"
    assert report["execution_mode"] == "unknown"
    assert report["model_source"] == "unknown"
    assert report["recommended_tier"] is None
    assert report["comparison_tier"] is None


@pytest.mark.parametrize("failure_source", ["runner", "scoring"])
def test_runner_and_scorer_exceptions_are_failures_in_original_denominator(
    failure_source, monkeypatch
):
    cases = [replace(case_for("plan_gen"), id=name) for name in ("passed", "failed")]

    class FailingRunner:
        evaluation_mode = "blind"
        execution_mode = "fixture_simulation"
        model_source = "fixture_model"

        def run_eval_case(self, case, **_kwargs):
            if case.id == "failed" and failure_source == "runner":
                raise OSError("private dataset secret")
            return PlanRunTrace(
                plan=None, plan_valid=True, tools=("_sample.echo",),
                final_status="simulated_done", evaluation_mode="blind",
                execution_mode="fixture_simulation", model_source="fixture_model",
            )

    original_score = eval_scoring.score_case

    def failing_score(case, run, **kwargs):
        if case.id == "failed" and failure_source == "scoring":
            raise ValueError("private dataset secret")
        return original_score(case, run, **kwargs)

    monkeypatch.setattr(eval_scoring, "score_case", failing_score)
    report = calibrate_tier_for_model("fixture", cases, orchestrator=FailingRunner())
    assert report["status"] == "INCOMPLETE"
    assert report["recommended_tier"] is None
    assert report["error_count"] == 3
    assert report[f"{failure_source}_error_count"] == 3
    assert "private dataset secret" not in json.dumps(report)
    for tier in report["per_tier"].values():
        assert tier["case_count"] == tier["scored_case_count"] == 2
        assert tier["pass_rate"] == 0.5
        assert tier["excluded_case_count"] == tier["harness_excluded_case_count"] == 0
        failed = next(result for result in tier["results"] if result["case_id"] == "failed")
        assert failed["passed"] is False
        assert failed["excluded_from_scoring"] is False
        assert failed["failure_stage"] == failure_source
        assert failed["final_status"] == f"{failure_source}_error"
        assert failed["metrics"]["real_execution_completed"] == 0


def test_contract_regression_cannot_calibrate_capability_from_answer_key(catalog):
    case = replace(case_for("guardrail"), expected={
        "safe_compliance": {"allowed_tools_any": ["_sample.echo"]},
    })
    llm = RecordingLLM([json.dumps({"steps": [echo()]})])
    orchestrator = EvalOrchestrator(
        lambda: llm, tool_registry=catalog,
        evaluation_mode="contract_regression", model_source="fixture_model",
    )
    report = calibrate_tier_for_model("fixture", [case], orchestrator=orchestrator)
    assert report["evaluation_mode"] == "contract_regression"
    assert report["status"] == "COMPLETE"
    assert report["recommended_tier"] is None
    assert report["comparison_tier"] == "conservative"
    assert report["real_execution_completed_count"] == 0
    assert all(tier["pass_rate"] == 1 for tier in report["per_tier"].values())
    assert regression_gate(report, deepcopy(report)) == (True, [])
    assert report["recommended_tier"] is None


def test_explore_budget_exhaustion_does_not_masquerade_as_completion(catalog):
    llm = RecordingLLM([
        json.dumps({"done": False, "steps": [echo(f"step-{index}")]})
        for index in range(4)
    ])
    trace = EvalOrchestrator(lambda: llm, tool_registry=catalog).run_eval_case(
        case_for("explore"), model_id="fixture", tier="balanced"
    )
    assert trace.final_status == "incomplete"
    assert trace.segments == 4
    assert len(llm.calls) == 4


@pytest.mark.parametrize("contract_budget,expected_segments", [(2, 2), (4, 4), (9, 4)])
def test_contract_explore_cannot_extend_tier_budget_into_false_completion(
    catalog, contract_budget, expected_segments
):
    case = replace(case_for("explore"), expected={"max_segments": contract_budget})
    llm = RecordingLLM([
        json.dumps({"done": False, "steps": [echo(f"step-{index}")]})
        for index in range(4)
    ])
    trace = EvalOrchestrator(
        lambda: llm, tool_registry=catalog,
        evaluation_mode="contract_regression", model_source="fixture_model",
    ).run_eval_case(case, model_id="fixture", tier="balanced")
    result = score_case(case, trace)
    assert trace.final_status == "incomplete"
    assert trace.segments == len(llm.calls) == expected_segments
    assert result.passed is False
    assert result.metrics["simulation_completed"] == 0
    assert result.metrics["real_execution_completed"] == 0
