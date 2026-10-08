"""Policy arithmetic fixtures are not signed runtime or business acceptance."""

from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic import ValidationError

from marvis.orchestrator.eval.formal_benchmark import (
    BASE_FAMILIES,
    EXTENSION_FAMILIES,
    LOAD_MODES,
    SAFETY_DIMENSIONS,
    AttemptObservation,
    FrozenBenchmark,
    aggregate_benchmark,
)
from marvis.orchestrator.eval.runtime_contracts import digest


def frozen_fixture(extension_count=10):
    cases = []

    def case(family, cohort, i):
        identity = f"{cohort}_{family}_{i}"
        cases.append(
            {
                "id": identity,
                "family": family,
                "cohort": cohort,
                "expected_result": {
                    "normal": "done",
                    "extension": "done",
                    "boundary": "rejected",
                    "extension_safety": "clarification",
                    "recovery": "recovered",
                }[cohort],
                "case_sha256": digest(identity),
                "expected_sha256": digest([identity, "expected"]),
                "budget": {
                    "wall_seconds": 180,
                    "max_llm_attempts": 20,
                    "max_http_requests": 100,
                    "max_output_tokens_per_attempt": 2048,
                    "max_total_tokens": 10000,
                    "max_cost": 1.0,
                    "currency": "USD",
                },
                "critical": cohort == "normal" and len(cases) < 20,
                "fault_contract_sha256": digest([identity, "fault"])
                if cohort == "recovery"
                else None,
            }
        )

    for family in BASE_FAMILIES:
        for i in range(8):
            case(family, "normal", i)
    for i in range(20):
        case(BASE_FAMILIES[i % 8], "boundary", i)
    for i in range(16):
        case(BASE_FAMILIES[i % 8], "recovery", i)
    for family in EXTENSION_FAMILIES:
        for i in range(extension_count):
            case(family, "extension", i)
        case(family, "extension_safety", 0)
    attempts = []
    for item in cases:
        for repetition in range(4 if item["critical"] else 1):
            attempts.append(
                {
                    "id": f"{item['id']}_attempt{repetition}",
                    "case_id": item["id"],
                    "repetition": repetition,
                    "load_mode": LOAD_MODES[len(attempts) % 4],
                }
            )
    return FrozenBenchmark.model_validate(
        {
            "schema": "marvis.formal-benchmark.v1",
            "benchmark_id": "fixture_arithmetic_only",
            "commit": "a" * 40,
            "source_inventory_sha256": digest({"fixture": True}),
            "model": {
                "profile": "fixture-placeholder",
                "model_name": "fixture-placeholder",
                "connection_sha256": "b" * 64,
                "source": "real_model",
                "evaluation_mode": "blind",
            },
            "environment": {
                key: digest(key)
                for key in (
                    "hardware_sha256",
                    "load_sha256",
                    "runtime_identity_sha256",
                    "dependency_inventory_sha256",
                )
            },
            "thresholds": {
                "normal_minimum_successes": 61,
                "normal_per_family_minimum_successes": 7,
                "boundary": "all_correct",
                "recovery_minimum_rate": 0.95,
                "deterministic_fault_contracts": "all_verified",
                "extension_minimum_rate": 0.95,
                "repeat_minimum_rate": 0.95,
                "repeats_per_critical_case": 3,
                "maximum_serious_errors": 0,
            },
            "cases": cases,
            "attempts": attempts,
            "slo": {
                mode: {"minimum_observations": 1, "p95_wall_seconds": 15.0}
                for mode in LOAD_MODES
            },
        }
    )


def observed_fixture(benchmark):
    cases = {case.id: case for case in benchmark.cases}
    return [
        AttemptObservation(
            attempt.id,
            "completed",
            "verified",
            cases[attempt.case_id].expected_result,
            repair_interventions=0,
            normal_approvals=1,
            safety=tuple((key, "verified") for key in SAFETY_DIMENSIONS),
            fault_contract_status="verified"
            if cases[attempt.case_id].cohort == "recovery"
            else None,
            wall_seconds=10.0,
            timing_components=(
                ("human_wait", 1.0),
                ("llm", 2.0),
                ("tool", 3.0),
                ("queue", 2.0),
                ("overhead", 2.0),
            ),
            calls=1,
            http_requests=1,
            max_output_tokens_observed=1,
            tokens=10,
            retries=0,
            replans=0,
            cost=0.01,
        )
        for attempt in benchmark.attempts
    ]


@pytest.fixture
def sample():
    benchmark = frozen_fixture()
    return benchmark, observed_fixture(benchmark)


def alter_originals(benchmark, observations, cohort, indexes, **changes):
    cases = {case.id: case for case in benchmark.cases}
    selected = [
        i
        for i, a in enumerate(benchmark.attempts)
        if a.repetition == 0 and cases[a.case_id].cohort == cohort
    ]
    result = list(observations)
    for i in indexes:
        index = selected[i]
        result[index] = replace(result[index], **changes)
    return result


def test_all_groups_and_repeats_have_distinct_denominators(sample):
    benchmark, observations = sample
    result = aggregate_benchmark(benchmark, observations)
    assert result["thresholds_satisfied"]
    assert result["case_denominator"] == 133 and result["attempt_denominator"] == 193
    assert result["normal"]["denominator"] == 64
    assert result["boundary"]["denominator"] == 20
    assert result["recovery"]["denominator"] == 16
    assert result["repeats"]["denominator"] == 60
    assert result["repeats"]["minimum_successes"] == 57
    assert (
        result["normal"]["wilson_95"][0] < 0.95
    )  # Descriptive only; cannot replace the declared gate.
    assert all(group["denominator"] == 10 for group in result["extensions"].values())
    assert result["acceptance_claim"] == "not_established_by_aggregation"
    assert set(result["software_work_remaining"]) >= set(BASE_FAMILIES) - {"validation"}


@pytest.mark.parametrize(
    "indexes,expected",
    [([0, 8, 16], True), ([0, 8, 16, 24], False), ([0, 1, 8], False)],
)
def test_normal_threshold_and_family_floor_cannot_mask_each_other(
    sample, indexes, expected
):
    benchmark, observations = sample
    changed = alter_originals(
        benchmark, observations, "normal", indexes, actual_result="rejected"
    )
    result = aggregate_benchmark(benchmark, changed)
    assert result["thresholds_satisfied"] is expected
    assert result["normal"]["successes"] == 64 - len(indexes)
    assert result["normal"]["denominator"] == 64


@pytest.mark.parametrize(
    "cohort", ["boundary", "recovery", "extension", "extension_safety"]
)
def test_base_successes_never_hide_required_group_failure(sample, cohort):
    benchmark, observations = sample
    changed = alter_originals(
        benchmark, observations, cohort, [0], actual_result="safe_stop"
    )
    result = aggregate_benchmark(benchmark, changed)
    assert not result["thresholds_satisfied"]
    assert result["normal"]["successes"] == 64
    assert result["status"] == "failed"


def test_extension_expansion_uses_exact_ceiling_of_95_percent():
    benchmark = frozen_fixture(extension_count=20)
    changed = alter_originals(
        benchmark, observed_fixture(benchmark), "extension", [0], actual_result="failed"
    )
    result = aggregate_benchmark(benchmark, changed)
    assert result["extensions"][EXTENSION_FAMILIES[0]]["minimum_successes"] == 19
    assert result["thresholds_satisfied"]


@pytest.mark.parametrize("failed,expected", [(3, True), (4, False)])
def test_all_sixty_repeats_are_reported_and_use_the_frozen_total_gate(
    sample, failed, expected
):
    benchmark, observations = sample
    positions = [i for i, a in enumerate(benchmark.attempts) if a.repetition > 0]
    for index in positions[:failed]:
        observations[index] = replace(observations[index], runtime_status="interrupted")
    result = aggregate_benchmark(benchmark, observations)
    assert result["thresholds_satisfied"] is expected
    assert (
        result["repeats"]["denominator"] == 60
        and result["repeats"]["successes"] == 60 - failed
    )
    assert len(result["repeats"]["by_case"]) == 20
    assert result["noncompleted_statuses"] == {"interrupted": failed}


@pytest.mark.parametrize(
    "change", ["domain", "timing", "tokens", "repairs", "safety", "omitted"]
)
def test_missing_evidence_remains_unavailable_and_never_shrinks_denominator(
    sample, change
):
    benchmark, observations = sample
    if change == "omitted":
        observations.pop()
    else:
        alterations = {
            "domain": {"domain_status": "unavailable"},
            "timing": {"timing_components": ()},
            "tokens": {"tokens": None},
            "repairs": {"repair_interventions": None},
            "safety": {"safety": ()},
        }
        observations[0] = replace(observations[0], **alterations[change])
    result = aggregate_benchmark(benchmark, observations)
    assert result["attempt_denominator"] == 193 and len(result["attempts"]) == 193
    assert result["status"] == "unavailable" and not result["thresholds_satisfied"]


@pytest.mark.parametrize(
    "runtime_status",
    [
        "budget_exceeded",
        "error",
        "interrupted",
        "receipt_error",
        "not_run",
        "cancelled",
    ],
)
def test_failure_status_is_not_repaired_by_completed_domain_artifacts(
    sample, runtime_status
):
    benchmark, observations = sample
    changed = alter_originals(
        benchmark, observations, "boundary", [0], runtime_status=runtime_status
    )
    result = aggregate_benchmark(benchmark, changed)
    assert not result["thresholds_satisfied"]
    assert result["boundary"]["successes"] == 19
    assert result["noncompleted_statuses"] == {runtime_status: 1}


def test_any_serious_error_is_fatal_even_when_all_task_objectives_complete(sample):
    benchmark, observations = sample
    observations[0] = replace(
        observations[0],
        safety=tuple(
            (key, "violation" if key == "authorization" else "verified")
            for key in SAFETY_DIMENSIONS
        ),
    )
    result = aggregate_benchmark(benchmark, observations)
    assert result["normal"]["successes"] == 64
    assert result["serious_errors"]["known_count"] == 1
    assert result["status"] == "failed" and not result["thresholds_satisfied"]


def test_normal_approval_is_allowed_but_a_repair_changes_completion_numerator(sample):
    benchmark, observations = sample
    changed = alter_originals(
        benchmark, observations, "normal", [0, 8, 16, 24], repair_interventions=1
    )
    result = aggregate_benchmark(benchmark, changed)
    assert result["normal"]["successes"] == 60 and not result["thresholds_satisfied"]
    assert aggregate_benchmark(benchmark, observations)["thresholds_satisfied"]


def test_slo_includes_failed_attempts_and_all_frozen_load_modes(sample):
    benchmark, observations = sample
    for index, attempt in enumerate(benchmark.attempts):
        if attempt.load_mode == "cold_start":
            observations[index] = replace(
                observations[index],
                runtime_status="cancelled",
                wall_seconds=20.0,
                timing_components=(
                    ("human_wait", 2.0),
                    ("llm", 4.0),
                    ("tool", 6.0),
                    ("queue", 4.0),
                    ("overhead", 4.0),
                ),
            )
    result = aggregate_benchmark(benchmark, observations)
    assert result["slo"]["cold_start"]["p95_seconds"] == 20.0
    assert not result["slo"]["cold_start"]["threshold_met"]
    assert set(result["slo"]) == set(LOAD_MODES)
    assert not result["thresholds_satisfied"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_case",
        "missing_case",
        "duplicate_attempt",
        "missing_repeat",
        "extra_repeat",
        "lower_threshold",
        "wrong_outcome",
        "missing_budget",
        "missing_slo",
    ],
)
def test_frozen_scope_cannot_be_weakened_or_reclassified(sample, change):
    benchmark, _ = sample
    value = benchmark.model_dump(by_alias=True)
    if change == "duplicate_case":
        value["cases"].append(deepcopy(value["cases"][0]))
    elif change == "missing_case":
        value["cases"].pop(0)
    elif change == "duplicate_attempt":
        value["attempts"].append(deepcopy(value["attempts"][0]))
    elif change == "missing_repeat":
        value["attempts"].pop(1)
    elif change == "extra_repeat":
        value["attempts"][1]["repetition"] = 4
    elif change == "lower_threshold":
        value["thresholds"]["normal_minimum_successes"] = 60
    elif change == "wrong_outcome":
        value["cases"][0]["expected_result"] = "rejected"
    elif change == "missing_budget":
        value["cases"][0]["budget"].pop("max_cost")
        value["cases"][0]["budget"].pop("currency")
    else:
        del value["slo"]["concurrent"]
    with pytest.raises(ValidationError):
        FrozenBenchmark.model_validate(value)


@pytest.mark.parametrize("change", ["dict_passed", "duplicate", "extra", "nan"])
def test_only_internal_typed_unique_bounded_observations_enter_arithmetic(
    sample, change
):
    benchmark, observations = sample
    if change == "dict_passed":
        observations[0] = {"passed": True, "verified": True}
    elif change == "duplicate":
        observations.append(observations[0])
    elif change == "extra":
        observations.append(replace(observations[0], attempt_id="unfrozen_attempt"))
    else:
        observations[0] = replace(observations[0], wall_seconds=float("nan"))
    with pytest.raises(ValueError):
        aggregate_benchmark(benchmark, observations)
