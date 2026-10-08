"""Fixed formal adapter fixtures; none create real institutional trust."""

from copy import deepcopy
import json
import shutil
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.acceptance_trust import TrustInputError
from marvis.orchestrator.eval.formal_acceptance import (
    _check_case_definition,
    _domain,
    _labeling_domain,
    _expected_validation,
    _observe_attempt,
    _telemetry,
    _validation_domain,
)
from marvis.orchestrator.eval.formal_benchmark import (
    FormalCase,
    FrozenBenchmark,
    SAFETY_DIMENSIONS,
)
from marvis.orchestrator.eval.runtime_contracts import RuntimeAction, RuntimeCase, digest
from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
from marvis.orchestrator.eval.runtime_scoring import ExpectedCase, ExpectedSuite
from test_acceptance_trust import (
    bundle as bundle,
    check,
    configure_runtime_bundle,
    put,
    sign_entries,
    update_record,
)
from test_formal_benchmark import frozen_fixture
from test_runtime_archive_validation import actual_archive as actual_archive
from test_runtime_archive_labeling import label_archive as label_archive


def configure_formal(bundle, value=None):
    value = value or frozen_fixture().model_dump(by_alias=True)
    value["commit"] = bundle.config["commit"]
    value["source_inventory_sha256"] = digest(bundle.config["source_inventory"])
    benchmark = FrozenBenchmark.model_validate(value)
    reference = put(bundle.evidence, "formal-policy.json", value)
    bundle.finding["required_evidence_tiers"] = ["A"]
    bundle.spec["findings"][0]["required_evidence_tiers"] = ["A"]
    bundle.spec_ref = put(bundle.trust, "spec.json", bundle.spec)
    bundle.config["spec_sha256"] = bundle.spec_ref["sha256"]
    bundle.record.update(
        run_id=benchmark.benchmark_id,
        evidence_tier="A",
        environment_kind="local_reference",
        source_kind="deidentified_historical",
        real_executor={"id": "executor-person"},
        model_profile=benchmark.model.profile,
        model_source="real_model",
        evaluation_mode="blind",
        passed=True,
        verified=True,
    )
    bundle.record["input_hashes"][reference["path"]] = reference["sha256"]
    bundle.record["tool_receipts"] = bundle.record["evidence_paths"]
    update_record(bundle)
    for authority in bundle.config["authorities"].values():
        authority.update(
            tiers=["A"],
            environments=["local_reference"],
            sources=["deidentified_historical"],
        )
    bundle.config["records"][0].update(
        adapter="formal_benchmark.v1", inputs={"benchmark": reference, "runs": []}
    )
    bundle.execution = {
        "benchmark_sha256": reference["sha256"],
        "planned_attempt_ids": [a.id for a in benchmark.attempts],
        "model": benchmark.model.model_dump(),
        "environment": benchmark.environment.model_dump(),
        "observations": [],
    }
    bundle.review["coverage"] = [
        "frozen_criteria",
        "complete_run_denominator",
        "narrative_semantics",
        "blind_real_model_execution",
        "frozen_before_execution",
        "expected_answers_isolated",
        "fixed_hardware_load",
        "repair_classification",
    ]
    sign_entries(bundle)
    return bundle, benchmark


@pytest.fixture
def formal_bundle(bundle):
    return configure_formal(bundle)


def test_signed_complete_policy_without_originals_keeps_all_193_attempts_unavailable(
    formal_bundle,
):
    bundle, _ = formal_bundle
    result = check(bundle)
    assert not result["complete"]
    facts = result["trusted_environment"]["records"][0]["observed"]
    assert facts["denominator"] == 193 and facts["observed_attempts"] == 0
    assert facts["passed"] == 0 and len(facts["attempts"]) == 193
    assert (
        facts["normal"]["denominator"] == 64 and facts["repeats"]["denominator"] == 60
    )
    assert (
        facts["software_work_remaining"]["modeling"]
        == "deterministic_family_revalidator_not_implemented"
    )
    assert (
        facts["software_work_remaining"]["collections"]
        == "deterministic_family_revalidator_not_implemented"
    )
    assert facts["status"] == "unavailable"


@pytest.mark.parametrize(
    "attack",
    ["model", "environment", "attempts", "benchmark", "source", "candidate_passed"],
)
def test_external_signature_does_not_erase_frozen_context_mismatch(
    formal_bundle, attack
):
    bundle, _ = formal_bundle
    if attack == "model":
        bundle.execution["model"]["profile"] = "another-profile"
    elif attack == "environment":
        bundle.execution["environment"]["hardware_sha256"] = "f" * 64
    elif attack == "attempts":
        bundle.execution["planned_attempt_ids"].pop()
    elif attack == "benchmark":
        bundle.execution["benchmark_sha256"] = "f" * 64
    elif attack == "source":
        value = json.loads((bundle.evidence / "formal-policy.json").read_bytes())
        value["source_inventory_sha256"] = "f" * 64
        reference = put(bundle.evidence, "formal-policy.json", value)
        bundle.record["input_hashes"][reference["path"]] = reference["sha256"]
        bundle.config["records"][0]["inputs"]["benchmark"] = reference
        bundle.execution["benchmark_sha256"] = reference["sha256"]
        update_record(bundle)
    else:
        bundle.execution["passed"] = True
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    facts = result["trusted_environment"]["records"][0]["observed"]
    assert facts["denominator"] == 193 and facts["passed"] == 0
    assert facts["reason"] in {
        "formal_source_binding_mismatch",
        "formal_executor_context_mismatch",
        "invalid_trust_schema",
    }


@pytest.mark.parametrize(
    "attack", ["duplicate", "unfrozen", "repeat_reuses_carrier", "unsigned_observation"]
)
def test_attempt_inventory_cannot_reuse_or_rename_a_successful_carrier(
    formal_bundle, attack
):
    bundle, benchmark = formal_bundle
    first, repeat = benchmark.attempts[:2]
    assignment = {
        "attempt_id": first.id,
        "case_id": first.case_id,
        "domain_evidence": None,
        "telemetry": None,
    }
    entry = {
        "run_id": "not-executed-fixture",
        "run_dir": "nonexistent",
        "final_manifest_sha256": "a" * 64,
        "cases_file": {},
        "expected_file": {},
        "price_book": None,
        "assignments": [assignment],
    }
    if attack == "duplicate":
        entry["assignments"].append(deepcopy(assignment))
    elif attack == "unfrozen":
        assignment["attempt_id"] = "outside-frozen-inventory"
    elif attack == "repeat_reuses_carrier":
        entry["assignments"].append({**assignment, "attempt_id": repeat.id})
    bundle.config["records"][0]["inputs"]["runs"] = [entry]
    if attack != "unsigned_observation":
        bundle.execution["observations"] = [
            {
                "run_id": entry["run_id"],
                "case_id": a["case_id"],
                "attempt_id": a["attempt_id"],
                "telemetry_sha256": None,
            }
            for a in entry["assignments"]
        ]
    sign_entries(bundle)
    result = check(bundle)
    assert not result["complete"]
    observed = result["trusted_environment"]["records"][0]["observed"]
    assert observed["denominator"] == 193 and not observed["successful"]
    assert observed["reason"] in {
        "formal_attempt_reassigned_or_duplicated",
        "same_runtime_case_cannot_count_twice",
        "formal_executor_observation_inventory_mismatch",
    }


def test_other_families_never_inherit_validation_recomputation():
    benchmark = frozen_fixture()
    expected = ExpectedCase(
        result="done",
        assertions=[{"kind": "http_status", "stage": "done", "value": 200}],
    )
    for family in {case.family for case in benchmark.cases} - {"validation"}:
        case = next(case for case in benchmark.cases if case.family == family)
        # Deliberately unusable paths prove no generic validation reader runs.
        result, reason = _validation_domain(
            case, expected, {"candidate_passed": True}, None, None, None
        )
        assert (
            result is None
            and reason == "deterministic_family_revalidator_not_implemented"
        )
    for cohort in ("boundary", "recovery"):
        case = next(
            case
            for case in benchmark.cases
            if case.family == "validation" and case.cohort == cohort
        )
        assert (
            _validation_domain(case, expected, None, None, None, None)[1]
            == "deterministic_boundary_or_recovery_revalidator_not_implemented"
        )


@pytest.mark.parametrize("currency", ["USD", "EUR"])
def test_usage_cost_cannot_be_compared_in_another_currency(currency):
    benchmark = frozen_fixture()
    template = next(c for c in benchmark.cases if c.family == "modeling" and c.cohort == "normal")
    definition = RuntimeCase(
        id=template.id, revision="fixture-only", family="modeling",
        case_set="independently_held_hidden_acceptance", task={"task_type": "modeling"},
        materials=[{"path": "fixture.csv", "sha256": "a" * 64,
                    "source_kind": "deidentified_historical"}],
        budget=template.budget, business_constraints_source="fixture-only",
    )
    expected = ExpectedCase(result="done", assertions=[{"kind": "http_status", "stage": "done", "value": 200}])
    case = FormalCase.model_validate({**template.model_dump(),
        "case_sha256": digest(definition.model_dump()), "expected_sha256": digest(expected.model_dump())})
    attempt = next(a for a in benchmark.attempts if a.case_id == case.id)
    events = [
        {"event": "started", "attempt_id": "one", "logical_call_id": "one", "attempt": 1,
         "model_name": benchmark.model.model_name, "model_id": benchmark.model.profile, "trace_complete": True},
        {"event": "finished", "attempt_id": "one", "prompt_tokens": 100, "completion_tokens": 20,
         "usage_final": True},
    ]
    run = {"executions": {case.id: {"runtime_status": "completed", "duration_ms": 1000,
        "family": case.family, "scenario": "normal", "case_set": definition.case_set,
        "budget": case.budget.model_dump(), "http_events": []}},
        "attempts": {case.id: events}, "scores": {case.id: {"passed": True}}}
    price = json.dumps({"model_name": benchmark.model.model_name, "provider": "fixture",
        "currency": currency, "source": "fixture", "effective_date": "2026-10-06",
        "prompt_per_million": 1.0, "completion_per_million": 2.0}).encode()
    observed, _ = _observe_attempt(attempt, case, {"telemetry": None, "domain_evidence": None},
        definition, expected, None, None, run, price, benchmark)
    if currency == "EUR":
        assert observed.cost is None and observed.reason == "formal_cost_currency_mismatch"
    else:
        assert observed.cost == pytest.approx(0.00014)
        assert observed.reason == "deterministic_family_revalidator_not_implemented"
    assert observed.domain_status == "unavailable"


def test_expected_validation_uses_recomputed_values_instead_of_candidate_flags():
    checks = {
        key: {"status": "verified"}
        for key in (
            "materials",
            "pmml_rescoring",
            "metrics_recomputation",
            "confirmation_record_binding",
            "report_download_binding",
            "excel_values",
            "word_confirmed_text",
        )
    }
    checks["pmml_rescoring"]["rows_recomputed"] = 180
    recomputed = {"checks": checks, "supported_checks_verified": True}
    observed = {
        "execution": {
            "validation_pipeline": {
                "scored_rows": 999,
                "passed": True,
                "scoring_verified": False,
            }
        }
    }
    expected = ExpectedCase(
        result="done",
        assertions=[
            {
                "kind": "validation_pipeline_equals",
                "path": ["scored_rows"],
                "value": 180,
            }
        ],
    )
    assert _expected_validation(expected, observed, recomputed)
    bad = ExpectedCase(
        result="done",
        assertions=[
            {
                "kind": "validation_pipeline_equals",
                "path": ["scored_rows"],
                "value": 999,
            }
        ],
    )
    assert not _expected_validation(bad, observed, recomputed)
    unsupported = ExpectedCase(
        result="done",
        assertions=[{"kind": "output_equals", "path": ["passed"], "value": True}],
    )
    with pytest.raises(TrustInputError, match="predicate_unavailable"):
        _expected_validation(unsupported, observed, recomputed)


def _telemetry_fixture(tmp_path):
    benchmark = frozen_fixture()
    attempt = benchmark.attempts[0]
    actions = [
        RuntimeAction(kind="start_validation_agent", content="fixture initial request"),
        RuntimeAction(
            kind="confirm_current_validation_report", content="fixture normal approval"
        ),
    ]
    definition = SimpleNamespace(family="validation", actions=actions)
    run = {
        "manifest": {"run_id": "fixture-run"},
        "executions": {
            attempt.case_id: {"duration_ms": 10000, "human_interventions": 2}
        },
    }
    value = {
        "schema": "marvis.formal-process-observation.v1",
        "attempt_id": attempt.id,
        "case_id": attempt.case_id,
        "run_id": "fixture-run",
        "model": benchmark.model.model_dump(),
        "environment": benchmark.environment.model_dump(),
        "load_mode": attempt.load_mode,
        "segments": [
            {"phase": "llm", "start_ms": 0, "end_ms": 5000},
            {"phase": "tool", "start_ms": 5000, "end_ms": 10000},
        ],
        "human_action_sha256s": [digest(a.model_dump()) for a in actions],
        "replans": 0,
    }
    reference = put(tmp_path, "telemetry.json", value)
    record = {"artifact_hashes": {reference["path"]: reference["sha256"]}}
    return benchmark, attempt, definition, run, value, reference, record


@pytest.mark.parametrize("schema", ["marvis.runtime-process-observation.v1", "marvis.runtime-process-observation.v2", "marvis.runtime-process-observation.v3", "marvis.runtime-process-observation.v4", "marvis.runtime-process-observation.v5", "marvis.runtime-process-observation.v6"])
def test_native_material_selection_is_bound_and_is_not_a_repair(tmp_path, schema):
    from marvis.orchestrator.eval.runtime_process import material_selection_identity

    benchmark, attempt, definition, run, value, _, record = _telemetry_fixture(tmp_path)
    definition.model_dump = lambda: {"fixture_case": attempt.case_id}
    selection = material_selection_identity(definition)
    value["human_action_sha256s"].insert(0, selection)
    original = run["executions"][attempt.case_id]
    original["human_interventions"] = 3
    original["process_observation"] = {
        "schema": schema,
        "human_actions": [{"action_sha256": identity} for identity in value["human_action_sha256s"]],
    }
    if schema in {"marvis.runtime-process-observation.v4", "marvis.runtime-process-observation.v5", "marvis.runtime-process-observation.v6"}:
        original["process_observation"]["revisions"] = {"complete": True, "known_counts": {"structural_replan": 0}}
    reference = put(tmp_path, "telemetry.json", value)
    record["artifact_hashes"][reference["path"]] = reference["sha256"]
    result, _ = _telemetry(tmp_path, record, reference, benchmark, attempt, run, definition)
    assert result["repair_interventions"] == 0 and result["normal_approvals"] == 1
    value["human_action_sha256s"][1] = digest("replacement claiming no repair")
    reference = put(tmp_path, "telemetry.json", value)
    record["artifact_hashes"][reference["path"]] = reference["sha256"]
    with pytest.raises(TrustInputError, match="human_action_original_binding_mismatch"):
        _telemetry(tmp_path, record, reference, benchmark, attempt, run, definition)


@pytest.mark.parametrize("incomplete", [False, True])
def test_external_replan_count_cannot_override_native_revision_observation(tmp_path, incomplete):
    benchmark, attempt, definition, run, value, _, record = _telemetry_fixture(tmp_path)
    run["executions"][attempt.case_id]["process_observation"] = {
        "schema": "marvis.runtime-process-observation.v4",
        "human_actions": [{"action_sha256": identity} for identity in value["human_action_sha256s"]],
        "revisions": {"complete": not incomplete, "known_counts": {"structural_replan": 0}},
    }
    value["replans"] = 0 if incomplete else 1
    reference = put(tmp_path, "telemetry.json", value)
    record["artifact_hashes"][reference["path"]] = reference["sha256"]
    with pytest.raises(TrustInputError, match="replan_original_binding_mismatch"):
        _telemetry(tmp_path, record, reference, benchmark, attempt, run, definition)


def test_original_total_and_disjoint_components_and_normal_approval_are_separate(
    tmp_path,
):
    benchmark, attempt, definition, run, value, reference, record = _telemetry_fixture(
        tmp_path
    )
    result, reason = _telemetry(
        tmp_path, record, reference, benchmark, attempt, run, definition
    )
    assert reason is None and result["wall_seconds"] == 10
    assert result["repair_interventions"] == 0 and result["normal_approvals"] == 1
    assert dict(result["timing_components"]) == {
        "human_wait": 0,
        "llm": 5,
        "tool": 5,
        "queue": 0,
        "overhead": 0,
    }
    value["human_action_sha256s"][1] = digest("operator repair instruction")
    reference = put(tmp_path, "telemetry.json", value)
    record["artifact_hashes"][reference["path"]] = reference["sha256"]
    result, _ = _telemetry(
        tmp_path, record, reference, benchmark, attempt, run, definition
    )
    assert result["repair_interventions"] == 1 and result["normal_approvals"] == 0


@pytest.mark.parametrize(
    "attack",
    [
        "overlap",
        "gap",
        "short",
        "hardware",
        "load",
        "human_omitted",
        "candidate_passed",
    ],
)
def test_process_observations_must_cover_original_timing_and_frozen_environment(
    tmp_path, attack
):
    benchmark, attempt, definition, run, value, reference, record = _telemetry_fixture(
        tmp_path
    )
    if attack == "overlap":
        value["segments"][1]["start_ms"] = 4999
    elif attack == "gap":
        value["segments"][1]["start_ms"] = 5001
    elif attack == "short":
        value["segments"][1]["end_ms"] = 9999
    elif attack == "hardware":
        value["environment"]["hardware_sha256"] = "f" * 64
    elif attack == "load":
        value["load_mode"] = "unfrozen"
    elif attack == "human_omitted":
        value["human_action_sha256s"].pop()
    else:
        value["passed"] = True
    reference = put(tmp_path, "telemetry.json", value)
    record["artifact_hashes"][reference["path"]] = reference["sha256"]
    with pytest.raises(TrustInputError):
        _telemetry(tmp_path, record, reference, benchmark, attempt, run, definition)


def test_actual_http_archive_recomputes_only_supported_validation_scope(
    bundle, actual_archive
):
    configured = configure_runtime_bundle(bundle, actual_archive)
    archive, _, binding = actual_archive
    item = configured.config["records"][0]["inputs"]["cases"][0]
    run = verify_run_manifest(
        configured.evidence / "run",
        expected_manifest_sha256=configured.config["records"][0]["inputs"][
            "final_manifest_sha256"
        ],
    )
    expected = ExpectedSuite.model_validate_json(
        (archive.parents[2] / "suite/private/expected.json").read_bytes()
    ).cases[binding.case.id]
    template = next(
        case
        for case in frozen_fixture().cases
        if case.family == "validation" and case.cohort == "normal"
    )
    case = FormalCase.model_validate(
        {
            **template.model_dump(),
            "id": binding.case.id,
            "case_sha256": digest(binding.case.model_dump()),
            "expected_sha256": digest(expected.model_dump()),
        }
    )
    result, reason = _validation_domain(
        case, expected, item, configured.evidence, configured.record, run
    )
    assert (
        reason is None
        and result["domain_status"] == "verified"
        and result["actual_result"] == "done"
    )
    assert result["domain_checks"]["pmml_rescoring"]["rows_recomputed"] == 180
    safety = dict(result["safety"])
    assert safety["evidence_binding"] == safety["false_completion"] == "verified"
    assert {key for key, status in safety.items() if status == "unavailable"} == set(
        SAFETY_DIMENSIONS
    ) - {"evidence_binding", "false_completion"}


@pytest.mark.parametrize("archive_fixture,family", [("actual_archive", "validation"), ("label_archive", "labeling")])
def test_actual_fixture_model_cannot_be_upgraded_by_formal_record_or_signature(
    bundle, request, archive_fixture, family
):
    actual_archive = request.getfixturevalue(archive_fixture)
    configured = configure_runtime_bundle(bundle, actual_archive)
    archive, _, binding = actual_archive
    old_inputs = deepcopy(configured.config["records"][0]["inputs"])
    original = verify_run_manifest(
        configured.evidence / "run",
        expected_manifest_sha256=old_inputs["final_manifest_sha256"],
    )
    value = frozen_fixture().model_dump(by_alias=True)
    chosen = next(
        case
        for case in value["cases"]
        if case["family"] == family and case["cohort"] == "normal"
    )
    old_id = chosen["id"]
    chosen["id"] = binding.case.id
    for attempt in value["attempts"]:
        if attempt["case_id"] == old_id:
            attempt["case_id"] = binding.case.id
    frozen = original["manifest"]["frozen"]
    value["model"].update(
        profile=frozen["model_id"],
        model_name=frozen["model_name"],
        connection_sha256=frozen["model_connection_sha256"],
    )
    configured, benchmark = configure_formal(configured, value)
    case_ref = put(
        configured.evidence,
        "cases.json",
        (archive.parents[2] / "suite/cases.json").read_bytes(),
    )
    expected_ref = put(
        configured.evidence,
        "expected.json",
        (archive.parents[2] / "suite/private/expected.json").read_bytes(),
    )
    configured.record["input_hashes"].update(
        {
            case_ref["path"]: case_ref["sha256"],
            expected_ref["path"]: expected_ref["sha256"],
        }
    )
    attempt = next(
        a
        for a in benchmark.attempts
        if a.case_id == binding.case.id and a.repetition == 0
    )
    entry = {
        "run_id": binding.run_id,
        "run_dir": "run",
        "final_manifest_sha256": old_inputs["final_manifest_sha256"],
        "cases_file": case_ref,
        "expected_file": expected_ref,
        "price_book": None,
        "assignments": [
            {
                "attempt_id": attempt.id,
                "case_id": binding.case.id,
                "domain_evidence": old_inputs["cases"][0],
                "telemetry": None,
            }
        ],
    }
    configured.config["records"][0]["inputs"]["runs"] = [entry]
    configured.execution["observations"] = [
        {
            "attempt_id": attempt.id,
            "case_id": binding.case.id,
            "run_id": binding.run_id,
            "telemetry_sha256": None,
        }
    ]
    update_record(configured)
    sign_entries(configured)
    result = check(configured)
    assert not result["complete"]
    facts = result["trusted_environment"]["records"][0]["observed"]
    assert facts["denominator"] == 193 and facts["passed"] == 0
    observed = next(row for row in facts["attempts"] if row["attempt_id"] == attempt.id)
    assert observed["reason"] == "formal_runtime_model_mismatch"
    assert (
        observed["runtime_status"] == "completed"
        and observed["domain_status"] == "unavailable"
    )


@pytest.fixture
def formal_label_originals(label_archive, tmp_path):
    archive, sha, binding = label_archive
    shutil.copytree(archive, tmp_path / "archive")
    reference = put(tmp_path, "binding.json", binding.model_dump())
    public = archive.parents[2] / "public" / binding.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json((archive.parents[2] / "suite/private/expected.json").read_bytes()).cases[binding.case.id]
    template = next(case for case in frozen_fixture().cases if case.family == "labeling" and case.cohort == "normal")
    case = FormalCase.model_validate({**template.model_dump(), "id": binding.case.id,
        "case_sha256": digest(binding.case.model_dump()), "expected_sha256": digest(expected.model_dump())})
    return SimpleNamespace(case=case, expected=expected, binding=binding, root=tmp_path, run=run,
        item={"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": reference},
        record={"input_hashes": {reference["path"]: reference["sha256"]}})


def test_formal_label_domain_recomputes_originals_but_public_development_is_not_hidden_acceptance(formal_label_originals):
    f = formal_label_originals
    result, reason = _domain(f.case, f.expected, f.item, f.root, f.record, f.run)
    assert reason is None and result["domain_status"] == "verified" and result["actual_result"] == "done"
    assert result["domain_checks"]["label_recomputation"]["status"] == "verified"
    assert dict(result["safety"])["false_completion"] == "verified"
    assert {key for key, status in result["safety"] if status == "unavailable"} == set(SAFETY_DIMENSIONS) - {"evidence_binding", "false_completion"}
    with pytest.raises(TrustInputError, match="formal_hidden_family_mismatch"):
        _check_case_definition(f.case, f.expected, f.binding.case, f.run["executions"][f.case.id])


@pytest.mark.parametrize("mutation,reason", [
    ("plan", "final_label_plan_binding_mismatch"), ("step", "final_label_step_binding_mismatch"),
    ("download", "final_label_download_binding_mismatch"), ("record", "case_input_not_record_bound"),
    ("definition", "formal_domain_definition_mismatch"),
])
def test_formal_label_domain_requires_external_case_and_final_public_bindings(formal_label_originals, mutation, reason):
    f = formal_label_originals
    observed = f.run["executions"][f.case.id]
    if mutation == "plan":
        observed["execution"]["plans"][0]["status"] = "failed"
    elif mutation == "step":
        next(step for step in observed["execution"]["steps"] if step["id"] == f.binding.step_ids["labels"])["output_sha256"] = "a" * 64
    elif mutation == "download":
        observed["http_events"] = [event for event in observed["http_events"] if event.get("stage") != "download_labeling_dataset"]
    elif mutation == "record":
        f.record["input_hashes"].clear()
    else:
        f.case = f.case.model_copy(update={"case_sha256": "a" * 64})
    with pytest.raises(TrustInputError, match=reason):
        _domain(f.case, f.expected, f.item, f.root, f.record, f.run)


def test_formal_label_domain_rejects_unsupported_claims_and_other_cohorts(formal_label_originals):
    f = formal_label_originals
    expected = ExpectedCase(result="done", assertions=[{"kind": "output_equals", "tool": "labeling.define_label", "path": ["business_accepted"], "value": True}])
    with pytest.raises(TrustInputError, match="formal_labeling_predicate_unavailable"):
        _domain(f.case, expected, f.item, f.root, f.record, f.run)
    assert _labeling_domain(f.case.model_copy(update={"family": "validation"}), f.expected, None, None, None, None) == (None, "deterministic_family_revalidator_not_implemented")
    for cohort in ("boundary", "recovery"):
        assert _domain(f.case.model_copy(update={"cohort": cohort}), f.expected, None, None, None, None) == (None, "deterministic_boundary_or_recovery_revalidator_not_implemented")
