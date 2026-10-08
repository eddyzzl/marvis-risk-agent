"""The formal adapter binds feature originals without upgrading fixture trust."""
import json
import shutil
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.acceptance_trust import TrustInputError
from marvis.orchestrator.eval.formal_acceptance import _domain, _check_case_definition
from marvis.orchestrator.eval.formal_benchmark import FormalCase, SAFETY_DIMENSIONS, SOFTWARE_GAPS
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
from marvis.orchestrator.eval.runtime_scoring import ExpectedCase, ExpectedSuite
from test_acceptance_trust import put
from test_formal_benchmark import frozen_fixture
from test_runtime_feature_download import feature_download as feature_download
from test_runtime_archive_feature import feature_archive as feature_archive


@pytest.fixture
def formal_feature_originals(feature_archive, tmp_path):
    archive, sha, binding = feature_archive
    shutil.copytree(archive, tmp_path / "archive")
    reference = put(tmp_path, "binding.json", binding.model_dump())
    public = archive.parents[2] / "public" / binding.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json((archive.parents[2] / "suite/private/expected.json").read_bytes()).cases[binding.case.id]
    template = next(case for case in frozen_fixture().cases if case.family == "feature" and case.cohort == "normal")
    case = FormalCase.model_validate({**template.model_dump(), "id": binding.case.id,
        "case_sha256": digest(binding.case.model_dump()), "expected_sha256": digest(expected.model_dump())})
    return SimpleNamespace(case=case, expected=expected, binding=binding, root=tmp_path, run=run,
        item={"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": reference},
        record={"input_hashes": {reference["path"]: reference["sha256"]}})


def test_formal_feature_recomputes_domain_but_retains_synthetic_and_safety_boundaries(formal_feature_originals):
    f = formal_feature_originals
    result, reason = _domain(f.case, f.expected, f.item, f.root, f.record, f.run)
    assert reason is None and result["domain_status"] == "verified" and result["actual_result"] == "done", result
    assert result["domain_checks"]["workbook_cells"]["status"] == "verified"
    assert {key for key, status in result["safety"] if status == "unavailable"} == set(SAFETY_DIMENSIONS) - {"evidence_binding", "false_completion"}
    assert "optional_metrics_binning" in SOFTWARE_GAPS["feature"]
    with pytest.raises(TrustInputError, match="formal_hidden_family_mismatch"):
        _check_case_definition(f.case, f.expected, f.binding.case, f.run["executions"][f.case.id])


@pytest.mark.parametrize("mutation,reason", [
    ("plan", "final_feature_plan_binding_mismatch"), ("step", "final_feature_step_binding_mismatch"),
    ("download", "final_feature_download_binding_mismatch"), ("record", "case_input_not_record_bound"),
    ("definition", "formal_domain_definition_mismatch"),
])
def test_final_public_records_and_external_binding_required(formal_feature_originals, mutation, reason):
    f = formal_feature_originals
    observed = f.run["executions"][f.case.id]
    if mutation == "plan":
        observed["execution"]["plans"][0]["status"] = "failed"
    elif mutation == "step":
        next(s for s in observed["execution"]["steps"] if s["id"] == f.binding.step_ids["metrics"])["output_sha256"] = "a" * 64
    elif mutation == "download":
        observed["http_events"] = [e for e in observed["http_events"] if e.get("stage") != "download_feature_report"]
    elif mutation == "record":
        f.record["input_hashes"].clear()
    else:
        f.case = f.case.model_copy(update={"case_sha256": "a" * 64})
    with pytest.raises(TrustInputError, match=reason):
        _domain(f.case, f.expected, f.item, f.root, f.record, f.run)


def test_unimplemented_predicates_and_other_cohorts_do_not_inherit_normal_success(formal_feature_originals):
    f = formal_feature_originals
    expected = ExpectedCase(result="done", assertions=[{
        "kind": "output_equals", "tool": "feature.generate_feature_report", "path": ["business_accepted"], "value": True,
    }])
    with pytest.raises(TrustInputError, match="formal_feature_predicate_unavailable"):
        _domain(f.case, expected, f.item, f.root, f.record, f.run)
    for cohort in ("boundary", "recovery"):
        assert _domain(f.case.model_copy(update={"cohort": cohort}), f.expected, None, None, None, None) == (
            None, "deterministic_boundary_or_recovery_revalidator_not_implemented")


def test_expected_scalar_is_checked_against_recomputation(formal_feature_originals):
    f = formal_feature_originals
    expected = ExpectedCase(result="done", assertions=[{
        "kind": "output_equals", "tool": "feature.generate_feature_report", "path": ["feature_count"], "value": 3,
    }])
    result, reason = _domain(f.case, expected, f.item, f.root, f.record, f.run)
    assert reason is None and result["domain_status"] == "verified" and result["actual_result"] == "failed"
    assert "original_key_not_available" in json.dumps(result["unsupported"])
