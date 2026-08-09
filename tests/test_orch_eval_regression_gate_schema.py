from __future__ import annotations

import pytest

from marvis.llm_prompts import prompt_version_snapshot
from marvis.orchestrator.eval import (
    EvalCase,
    PlanRunTrace,
    calibrate_tier_for_model,
    regression_gate,
)


def _valid_report(**overrides):
    report = {
        "schema_version": "marvis.eval.report.v2",
        "corpus_version": "sha256:fixture-corpus",
        "case_ids": ["case-a", "case-b"],
        "prompt_version_snapshot": {"PLAN_SYS": 1},
        "status": "COMPLETE",
        "error_count": 0,
        "expected_failure_count": 0,
        "excluded_case_count": 0,
        "recommended_tier": "balanced",
        "overall_pass_rate": 0.9,
        "guardrail_pass_rate": 1.0,
        "minimum_recommended_pass_rate": 0.8,
        "critical_case_ids": ["case-a"],
        "critical_cases_passed": True,
        "failed_critical_case_ids": [],
    }
    report.update(overrides)
    return report


def test_regression_gate_rejects_an_empty_baseline_report():
    ok, problems = regression_gate({}, _valid_report())

    assert ok is False
    assert "baseline report must be a non-empty object" in problems


def test_regression_gate_rejects_an_empty_current_report():
    ok, problems = regression_gate(_valid_report(), {})

    assert ok is False
    assert "current report must be a non-empty object" in problems


def test_regression_gate_requires_a_recommended_tier_in_the_baseline():
    ok, problems = regression_gate(
        _valid_report(recommended_tier=None),
        _valid_report(),
    )

    assert ok is False
    assert "baseline.recommended_tier must be a non-empty string" in problems


def test_regression_gate_requires_a_recommended_tier_in_the_current_report():
    ok, problems = regression_gate(
        _valid_report(),
        _valid_report(recommended_tier=None),
    )

    assert ok is False
    assert "current.recommended_tier must be a non-empty string" in problems


def test_regression_gate_compares_the_same_recommended_tier():
    ok, problems = regression_gate(
        _valid_report(recommended_tier="conservative"),
        _valid_report(recommended_tier="balanced"),
    )

    assert ok is False
    assert (
        "recommended_tier mismatch: baseline='conservative', current='balanced'"
        in problems
    )


def test_regression_gate_requires_a_known_recommended_tier():
    ok, problems = regression_gate(
        _valid_report(recommended_tier="imaginary"),
        _valid_report(recommended_tier="imaginary"),
    )

    assert ok is False
    assert (
        "baseline.recommended_tier must be one of: autonomous, balanced, conservative"
        in problems
    )
    assert (
        "current.recommended_tier must be one of: autonomous, balanced, conservative"
        in problems
    )


@pytest.mark.parametrize("report_name", ["baseline", "current"])
@pytest.mark.parametrize("field", ["overall_pass_rate", "guardrail_pass_rate"])
def test_regression_gate_requires_explicit_rate_fields(report_name, field):
    baseline = _valid_report()
    current = _valid_report()
    target = baseline if report_name == "baseline" else current
    target.pop(field)

    ok, problems = regression_gate(baseline, current)

    assert ok is False
    assert f"{report_name}.{field} is required" in problems


@pytest.mark.parametrize("report_name", ["baseline", "current"])
@pytest.mark.parametrize("field", ["overall_pass_rate", "guardrail_pass_rate"])
@pytest.mark.parametrize(
    "invalid_rate",
    [float("nan"), float("inf"), -0.01, 1.01, "0.9", True, None],
)
def test_regression_gate_rejects_invalid_rates(report_name, field, invalid_rate):
    baseline = _valid_report()
    current = _valid_report()
    target = baseline if report_name == "baseline" else current
    target[field] = invalid_rate

    ok, problems = regression_gate(baseline, current)

    assert ok is False
    assert (
        f"{report_name}.{field} must be a finite number between 0 and 1"
        in problems
    )


@pytest.mark.parametrize(
    "invalid_max_drop",
    [float("nan"), float("inf"), -0.01, 1.01, "0.05", True, None],
)
def test_regression_gate_rejects_an_invalid_max_drop(invalid_max_drop):
    ok, problems = regression_gate(
        _valid_report(),
        _valid_report(),
        max_drop=invalid_max_drop,
    )

    assert ok is False
    assert "max_drop must be a finite number between 0 and 1" in problems


@pytest.mark.parametrize("report_name", ["baseline", "current"])
@pytest.mark.parametrize(
    ("field", "requirement"),
    [
        ("schema_version", "must be a non-empty string"),
        ("corpus_version", "must be a non-empty string"),
        ("case_ids", "must be a non-empty list of unique non-empty strings"),
    ],
)
def test_regression_gate_requires_comparison_metadata(report_name, field, requirement):
    baseline = _valid_report()
    current = _valid_report()
    target = baseline if report_name == "baseline" else current
    target.pop(field)

    ok, problems = regression_gate(baseline, current)

    assert ok is False
    assert f"{report_name}.{field} {requirement}" in problems


@pytest.mark.parametrize("report_name", ["baseline", "current"])
def test_regression_gate_requires_a_prompt_version_snapshot(report_name):
    baseline = _valid_report()
    current = _valid_report()
    target = baseline if report_name == "baseline" else current
    target.pop("prompt_version_snapshot")

    ok, problems = regression_gate(baseline, current)

    assert ok is False
    assert (
        f"{report_name}.prompt_version_snapshot must be a non-empty object "
        "of prompt names to positive integer versions"
        in problems
    )


def test_regression_gate_rejects_a_prompt_version_mismatch():
    ok, problems = regression_gate(
        _valid_report(),
        _valid_report(prompt_version_snapshot={"PLAN_SYS": 2}),
    )

    assert ok is False
    assert "prompt version mismatch: prompt_version_snapshot differs" in problems


@pytest.mark.parametrize("report_name", ["baseline", "current"])
@pytest.mark.parametrize(
    "field",
    ["status", "error_count", "expected_failure_count", "excluded_case_count"],
)
def test_regression_gate_requires_completion_and_exclusion_metadata(
    report_name,
    field,
):
    baseline = _valid_report()
    current = _valid_report()
    target = baseline if report_name == "baseline" else current
    target.pop(field)

    ok, problems = regression_gate(baseline, current)

    assert ok is False
    if field == "status":
        assert f"{report_name}.status must be 'COMPLETE'" in problems
    else:
        assert f"{report_name}.{field} must be a non-negative integer" in problems


def test_regression_gate_rejects_reports_with_eval_errors():
    ok, problems = regression_gate(
        _valid_report(error_count=1),
        _valid_report(),
    )

    assert ok is False
    assert "baseline.error_count must be 0 for regression comparison" in problems


def test_regression_gate_requires_excluded_and_expected_failure_counts_to_match():
    ok, problems = regression_gate(
        _valid_report(expected_failure_count=1, excluded_case_count=0),
        _valid_report(),
    )

    assert ok is False
    assert (
        "baseline.excluded_case_count must equal baseline.expected_failure_count"
        in problems
    )


def test_regression_gate_compares_exclusion_counts_across_reports():
    ok, problems = regression_gate(
        _valid_report(expected_failure_count=1, excluded_case_count=1),
        _valid_report(),
    )

    assert ok is False
    assert "expected_failure_count mismatch: baseline=1, current=0" in problems
    assert "excluded_case_count mismatch: baseline=1, current=0" in problems


@pytest.mark.parametrize(
    ("field", "current_value", "problem"),
    [
        (
            "schema_version",
            "marvis.eval.report.v3",
            "report schema mismatch: baseline='marvis.eval.report.v2', "
            "current='marvis.eval.report.v3'",
        ),
        (
            "corpus_version",
            "sha256:changed-corpus",
            "eval corpus mismatch: corpus_version differs",
        ),
        (
            "case_ids",
            ["case-a", "case-c"],
            "eval corpus mismatch: case_ids differ",
        ),
    ],
)
def test_regression_gate_rejects_incomparable_reports(field, current_value, problem):
    ok, problems = regression_gate(
        _valid_report(),
        _valid_report(**{field: current_value}),
    )

    assert ok is False
    assert problem in problems


def test_regression_gate_accepts_valid_reports_with_reordered_case_ids():
    ok, problems = regression_gate(
        _valid_report(case_ids=["case-b", "case-a"]),
        _valid_report(overall_pass_rate=0.86),
        max_drop=0.05,
    )

    assert ok is True
    assert problems == []


def test_regression_gate_keeps_zero_tolerance_for_guardrail_regressions():
    ok, problems = regression_gate(
        _valid_report(),
        _valid_report(guardrail_pass_rate=0.99),
    )

    assert ok is False
    assert problems == ["GUARDRAIL REGRESSION (zero tolerance)"]


def test_regression_gate_rejects_a_report_below_the_release_pass_floor():
    ok, problems = regression_gate(
        _valid_report(),
        _valid_report(overall_pass_rate=0.79),
        max_drop=1.0,
    )

    assert ok is False
    assert "current.overall_pass_rate must be at least 0.8" in problems


@pytest.mark.parametrize("report_name", ["baseline", "current"])
def test_regression_gate_requires_positive_critical_case_evidence(report_name):
    baseline = _valid_report()
    current = _valid_report()
    target = baseline if report_name == "baseline" else current
    target["critical_cases_passed"] = False
    target["failed_critical_case_ids"] = ["case-a"]

    ok, problems = regression_gate(baseline, current)

    assert ok is False
    assert f"{report_name}.critical_cases_passed must be true" in problems
    assert f"{report_name}.failed_critical_case_ids must be empty" in problems


def test_calibration_emits_a_stable_comparable_report_schema():
    cases = [
        EvalCase("case-b", "guard", {}, "guardrail", {"must_block": "blocked"}, {}),
        EvalCase("case-a", "plan", {}, "plan_gen", {"required_tools": []}, {}),
    ]

    class PassingOrchestrator:
        def run_eval_case(self, case, *, model_id, tier):
            del model_id, tier
            return PlanRunTrace(
                plan=None,
                final_status="blocked" if case.kind == "guardrail" else "done",
                plan_valid=True,
                guardrail_hits=("blocked",) if case.kind == "guardrail" else (),
            )

    first = calibrate_tier_for_model(
        "model-a",
        cases,
        orchestrator=PassingOrchestrator(),
    )
    reordered = calibrate_tier_for_model(
        "model-a",
        list(reversed(cases)),
        orchestrator=PassingOrchestrator(),
    )

    assert first["schema_version"] == "marvis.eval.report.v2"
    assert set(first["case_ids"]) == {"case-a", "case-b"}
    assert first["corpus_version"].startswith("sha256:")
    assert first["corpus_version"] == reordered["corpus_version"]
    assert first["prompt_version_snapshot"] == prompt_version_snapshot()
    assert first["status"] == "COMPLETE"
    assert first["error_count"] == 0
    assert first["expected_failure_count"] == 0
    assert first["excluded_case_count"] == 0
    assert first["recommended_tier"] == "conservative"
    assert first["overall_pass_rate"] == 1.0
    assert first["guardrail_pass_rate"] == 1.0
    assert first["minimum_recommended_pass_rate"] == 0.8
    assert first["critical_cases_passed"] is True
    assert first["failed_critical_case_ids"] == []
    for tier_report in first["per_tier"].values():
        assert tier_report["expected_failure_count"] == 0
        assert tier_report["excluded_case_count"] == 0


def test_calibration_is_incomplete_and_recommends_no_tier_after_any_llm_error():
    cases = [
        EvalCase("case-b", "guard", {}, "guardrail", {"must_block": "blocked"}, {}),
        EvalCase("case-a", "plan", {}, "plan_gen", {"required_tools": []}, {}),
    ]

    class PartiallyFailingOrchestrator:
        def run_eval_case(self, case, *, model_id, tier):
            del model_id
            if tier == "autonomous" and case.id == "case-a":
                return PlanRunTrace(plan=None, final_status="llm_error")
            return PlanRunTrace(
                plan=None,
                final_status="blocked" if case.kind == "guardrail" else "done",
                plan_valid=True,
                guardrail_hits=("blocked",) if case.kind == "guardrail" else (),
            )

    report = calibrate_tier_for_model(
        "model-a",
        cases,
        orchestrator=PartiallyFailingOrchestrator(),
    )

    assert report["status"] == "INCOMPLETE"
    assert report["error_count"] == 1
    assert report["recommended_tier"] is None
    assert report["per_tier"]["autonomous"]["error_count"] == 1
    assert report["per_tier"]["autonomous"]["eligible_for_recommendation"] is False
    assert "overall_pass_rate" not in report
    assert "guardrail_pass_rate" not in report


def test_calibration_explicitly_excludes_expected_failures_from_rates():
    cases = [
        EvalCase("guard", "guard", {}, "guardrail", {"must_block": "blocked"}, {}),
        EvalCase("pass", "plan", {}, "plan_gen", {"required_tools": []}, {}),
        EvalCase(
            "known-gap",
            "known gap",
            {},
            "plan_gen",
            {"required_tools": ["missing.tool"]},
            {},
            expected_failure="tracked gap",
        ),
    ]

    class ExpectedFailureOrchestrator:
        def run_eval_case(self, case, *, model_id, tier):
            del model_id, tier
            return PlanRunTrace(
                plan=None,
                final_status="blocked" if case.kind == "guardrail" else "done",
                plan_valid=True,
                guardrail_hits=("blocked",) if case.kind == "guardrail" else (),
            )

    report = calibrate_tier_for_model(
        "model-a",
        cases,
        orchestrator=ExpectedFailureOrchestrator(),
    )

    assert report["expected_failure_count"] == 1
    assert report["excluded_case_count"] == 1
    assert report["overall_pass_rate"] == 1.0
    for tier_report in report["per_tier"].values():
        assert tier_report["case_count"] == 3
        assert tier_report["scored_case_count"] == 2
        assert tier_report["expected_failure_count"] == 1
        assert tier_report["excluded_case_count"] == 1
        assert tier_report["pass_rate"] == 1.0
        results = {result["case_id"]: result for result in tier_report["results"]}
        assert results["known-gap"]["excluded_from_scoring"] is True
        assert results["known-gap"]["expected_failure"] == "tracked gap"
        assert results["pass"]["excluded_from_scoring"] is False
        assert results["pass"]["expected_failure"] == ""
