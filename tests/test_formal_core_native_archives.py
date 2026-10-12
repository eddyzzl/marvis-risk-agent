"""The installed formal dispatcher consumes actual Agent/worker originals."""
import shutil

import pytest

from marvis.orchestrator.eval.acceptance_trust import TrustInputError
from marvis.orchestrator.eval.formal_acceptance import _domain
from marvis.orchestrator.eval.formal_benchmark import FormalCase
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
from marvis.orchestrator.eval.runtime_scoring import ExpectedSuite
from tests.test_acceptance_trust import put
from tests.test_formal_benchmark import frozen_fixture
from tests.test_runtime_domain_archives import actual_archive as actual_archive
from tests.test_runtime_domain_archives import actual_refit_archive as actual_refit_archive


def formal_input(actual_archive, tmp_path):
    archive, sha, binding = actual_archive
    shutil.copytree(archive, tmp_path / "archive")
    ref = put(tmp_path, "binding.json", binding.model_dump())
    parent = archive.parents[2]
    public = parent / "public" / binding.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json((parent / "suite/private/expected.json").read_bytes()).cases[binding.case.id]
    template = next(c for c in frozen_fixture().cases if c.family == binding.case.family and c.cohort == "normal")
    case = FormalCase.model_validate({**template.model_dump(), "id": binding.case.id,
                                     "case_sha256": digest(binding.case.model_dump()),
                                     "expected_sha256": digest(expected.model_dump())})
    item = {"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": ref}
    return case, expected, item, {"input_hashes": {ref["path"]: ref["sha256"]}}, run


def test_formal_dispatch_recomputes_actual_core_originals_without_upgrading_original_expected(actual_archive, tmp_path):
    case, expected, item, record, run = formal_input(actual_archive, tmp_path)
    answer_hash = digest(expected.model_dump())
    result, reason = _domain(case, expected, item, tmp_path, record, run)
    assert reason is None and result["domain_status"] == "verified", result
    # The freshly generated public answer remains refit=False; native execution
    # and the independent dispatcher must honor that explicitly declared input.
    assert result["actual_result"] == "done"
    assert digest(expected.model_dump()) == answer_hash == case.expected_sha256
    if case.family != "strategy":
        scope = result["domain_checks"]["model_closure_fit_and_metrics"] if case.family == "modeling" else result["domain_checks"]["model_fit_and_baseline"]["fit_scope"]
        assert scope["fit_scope"] == "outer_train" and scope["fit_rows"] == 360
        assert scope["test_evaluation"] == "independent_heldout"
    assert dict(result["safety"])["authorization"] == "unavailable"
    assert dict(result["safety"])["point_in_time"] == "unavailable"


def test_formal_dispatch_recomputes_explicit_development_refit_without_rewriting_source(actual_refit_archive, tmp_path):
    case, expected, item, record, run = formal_input(actual_refit_archive, tmp_path)
    answer_hash = digest(expected.model_dump())
    result, reason = _domain(case, expected, item, tmp_path, record, run)
    assert reason is None and result["domain_status"] == "verified", result
    assert result["actual_result"] == "done"
    assert digest(expected.model_dump()) == answer_hash == case.expected_sha256
    scope = result["domain_checks"]["model_closure_fit_and_metrics"] if case.family == "modeling" else result["domain_checks"]["model_fit_and_baseline"]["fit_scope"]
    assert scope["fit_scope"] == "full_development" and scope["fit_rows"] == 480
    assert scope["test_evaluation"] == "unavailable_consumed_in_fit"
    assert dict(result["safety"])["authorization"] == "unavailable"
    assert dict(result["safety"])["point_in_time"] == "unavailable"


@pytest.mark.parametrize("field,value", [("output_sha256", "f" * 64), ("tool", "uninstalled.absent_tool")])
def test_formal_dispatch_rejects_final_public_step_identity_drift(actual_archive, tmp_path, field, value):
    case, expected, item, record, run = formal_input(actual_archive, tmp_path)
    binding = actual_archive[2]
    step_id = next(iter(binding.step_ids.values()))
    step = next(s for s in run["executions"][case.id]["execution"]["steps"] if s["id"] == step_id)
    step[field] = value
    with pytest.raises(TrustInputError, match="final_domain_step_binding_mismatch"):
        _domain(case, expected, item, tmp_path, record, run)
