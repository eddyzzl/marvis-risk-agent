"""Local actual carriers exercise formal seams, not real formal-A acceptance."""
from copy import deepcopy
import shutil
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.acceptance_trust import TrustInputError
from marvis.orchestrator.eval.formal_acceptance import _assignment_inventory, _domain
from marvis.orchestrator.eval.formal_benchmark import FormalCase, SOFTWARE_GAPS
from marvis.orchestrator.eval.formal_safety_adapters import future_fault_contract, merge_safety, retained_safety
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
from marvis.orchestrator.eval.runtime_scoring import ExpectedSuite
from tests.test_formal_benchmark import frozen_fixture
from tests.test_acceptance_trust import put
from tests.test_runtime_fault_archives import core_originals as core_originals
from tests.test_runtime_archive_join import _binding as join_binding


def inputs(core_originals, tmp_path, name):
    archive, sha, binding = core_originals[name]
    shutil.copytree(archive, tmp_path / "archive")
    ref = put(tmp_path, "safety-binding.json", binding.model_dump())
    public = archive.parents[2] / "public" / binding.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json((archive.parents[2] / "future-expected.json").read_bytes()).cases[binding.case.id]
    cohort = "recovery" if name == "discard" else "boundary" if name == "boundary" else "normal"
    template = next(case for case in frozen_fixture().cases if case.family == "data_processing" and case.cohort == cohort)
    case = FormalCase.model_validate({**template.model_dump(), "id": binding.case.id,
                                     "case_sha256": digest(binding.case.model_dump()), "expected_sha256": digest(expected.model_dump()),
                                     "fault_contract_sha256": digest(future_fault_contract(binding.case)) if cohort == "recovery" else None,
                                     "budget": binding.case.budget.model_dump()})
    item = {"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": ref}
    record = {"input_hashes": {ref["path"]: ref["sha256"]}}
    domain_item = None
    if name not in {"boundary", "reject"}:
        frozen = run["manifest"]["frozen"]
        report = {key: frozen[key] for key in ("cases_sha256", "expected_sha256", "source", "model_connection_sha256")}
        report["run_id"] = binding.run_id
        business = join_binding(tmp_path / "archive", report, run["executions"][binding.case.id], binding.case)
        business_ref = put(tmp_path, "join-binding.json", business.model_dump())
        record["input_hashes"][business_ref["path"]] = business_ref["sha256"]
        domain_item = {**item, "binding": business_ref}
    return SimpleNamespace(case=case, expected=expected, item=item, domain_item=domain_item, record=record, run=run, root=tmp_path, binding=binding)


def evaluate(value):
    return retained_safety(value.case, value.expected, value.item, value.root, value.record, value.run, domain_item=value.domain_item)


def test_original_assignment_wire_and_signed_observation_are_unchanged_without_safety():
    benchmark = frozen_fixture()
    attempt = benchmark.attempts[0]
    assignment = {"attempt_id": attempt.id, "case_id": attempt.case_id, "domain_evidence": None, "telemetry": None}
    entry = {"run_id": "local-fixture", "run_dir": "run", "final_manifest_sha256": "a" * 64,
             "cases_file": {}, "expected_file": {}, "price_book": None, "assignments": [assignment]}
    observations = [{"attempt_id": attempt.id, "case_id": attempt.case_id, "run_id": entry["run_id"], "telemetry_sha256": None}]
    original = deepcopy((entry, observations))
    assert _assignment_inventory({"runs": [entry]}, {"observations": observations}, benchmark) == [entry]
    assert (entry, observations) == original and "safety_evidence_sha256" not in observations[0]


@pytest.mark.parametrize("attack", ["unsigned", "altered", "pass_flag", "callback", "null"])
def test_new_safety_carrier_requires_exact_signed_observation_and_fixed_keys(attack):
    benchmark = frozen_fixture()
    attempt = benchmark.attempts[0]
    item = {"case_id": attempt.case_id, "archive": "archive", "manifest_sha256": "a" * 64,
            "binding": {"path": "binding.json", "sha256": "b" * 64}}
    assignment = {"attempt_id": attempt.id, "case_id": attempt.case_id, "domain_evidence": None, "telemetry": None, "safety_evidence": item}
    entry = {"run_id": "local-fixture", "run_dir": "run", "final_manifest_sha256": "a" * 64,
             "cases_file": {}, "expected_file": {}, "price_book": None, "assignments": [assignment]}
    observation = {"attempt_id": attempt.id, "case_id": attempt.case_id, "run_id": entry["run_id"], "telemetry_sha256": None, "safety_evidence_sha256": digest(item)}
    assert _assignment_inventory({"runs": [entry]}, {"observations": [observation]}, benchmark) == [entry]
    if attack == "unsigned":
        observation.pop("safety_evidence_sha256")
    elif attack == "altered":
        item["manifest_sha256"] = "c" * 64
    elif attack == "null":
        assignment["safety_evidence"] = None
    else:
        item["passed" if attack == "pass_flag" else "callback"] = True if attack == "pass_flag" else "arbitrary"
    with pytest.raises(TrustInputError):
        _assignment_inventory({"runs": [entry]}, {"observations": [observation]}, benchmark)


def test_normal_math_merges_native_safety_without_upgrading_missing_institution_authority(core_originals, tmp_path):
    value = inputs(core_originals, tmp_path, "stale")
    evidence = evaluate(value)
    assert evidence["reader_results"]["authorization"]["status"] == "verified"
    assert evidence["safety"]["authorization"] == "unavailable"
    assert evidence["safety"]["point_in_time"] == evidence["safety"]["duplicate_side_effects"] == "unavailable"
    domain, reason = _domain(value.case, value.expected, value.domain_item, value.root, value.record, value.run)
    assert reason is None and domain["domain_status"] == "verified"
    merged = merge_safety(domain, evidence)
    assert merged["actual_result"] == "done" and dict(merged["safety"])["authorization"] == "unavailable"
    assert dict(merged["safety"])["evidence_binding"] == "verified"


def test_boundary_uses_actual_409_zero_native_effect_and_both_original_expected_predicates(core_originals, tmp_path):
    value = inputs(core_originals, tmp_path, "boundary")
    evidence = evaluate(value)
    assert evidence["domain"]["domain_status"] == "verified" and evidence["domain"]["actual_result"] == "rejected"
    assert evidence["reader_results"]["boundary"]["values"]["boundaries"][0]["http_status"] == 409
    changed = value.expected.model_copy(update={"assertions": value.expected.assertions[:1]})
    case = value.case.model_copy(update={"expected_sha256": digest(changed.model_dump())})
    with pytest.raises(TrustInputError, match="outcome_closure_unavailable"):
        retained_safety(case, changed, value.item, value.root, value.record, value.run)


def test_recovery_requires_native_fault_binding_same_archive_independent_JOIN_and_original_expected(core_originals, tmp_path, monkeypatch):
    value = inputs(core_originals, tmp_path, "discard")
    from marvis.data.backend import DataBackend
    monkeypatch.setattr(DataBackend, "left_join", lambda *a, **k: pytest.fail("producer JOIN must not adjudicate original recovery"))
    evidence = evaluate(value)
    assert evidence["domain"]["domain_status"] == "verified" and evidence["domain"]["actual_result"] == "recovered"
    assert evidence["domain"]["fault_contract_status"] == "verified"
    assert evidence["domain"]["domain_checks"]["original_business_math"]["status"] == "verified"
    with pytest.raises(TrustInputError, match="frozen_fault_contract_mismatch"):
        retained_safety(value.case.model_copy(update={"fault_contract_sha256": "0" * 64}), value.expected, value.item, value.root, value.record, value.run, domain_item=value.domain_item)
    with pytest.raises(TrustInputError, match="original_join_math_unavailable"):
        retained_safety(value.case, value.expected, value.item, value.root, value.record, value.run)


@pytest.mark.parametrize("field,new_value", [("tool", "modeling.train_model"), ("status", "failed"), ("output_sha256", "0" * 64)])
def test_recovery_public_parent_steps_must_match_authenticated_JOIN_outputs(core_originals, tmp_path, field, new_value):
    value = inputs(core_originals, tmp_path, "discard")
    observed = value.run["executions"][value.case.id]
    parent = next(step for step in observed["execution"]["steps"] if step["tool"] == "data_ops.propose_join")
    parent[field] = new_value
    with pytest.raises(TrustInputError, match="recovery_public_step_mismatch"):
        evaluate(value)


@pytest.mark.parametrize("attack", ["task", "source", "family", "unbound", "case"])
def test_safety_binding_cannot_reassign_original_task_source_family_or_signed_reference(core_originals, tmp_path, attack):
    value = inputs(core_originals, tmp_path, "stale")
    binding = value.binding.model_dump()
    if attack == "task":
        binding["task_id"] = "different-task"
    elif attack == "source":
        binding["source"]["source_sha256"] = "0" * 64
    elif attack == "family":
        binding["case"]["family"] = "strategy"
    elif attack == "case":
        binding["case"]["actions"][1]["content"] = "a different complete human decision"
    else:
        value.record["input_hashes"].clear()
    if attack != "unbound":
        ref = put(tmp_path, "altered-safety-binding.json", binding)
        value.item["binding"] = ref
        value.record["input_hashes"][ref["path"]] = ref["sha256"]
    with pytest.raises(TrustInputError):
        evaluate(value)


def test_software_gap_inventory_reports_installed_scopes_and_explicit_unavailable_exits():
    assert "not_implemented" not in " ".join(SOFTWARE_GAPS.values())
    assert "future_fixed_stale_CAS_discard_readback" in SOFTWARE_GAPS["data_processing"]
    assert "institutional_authority" in SOFTWARE_GAPS["cross_workflow.safety"]
