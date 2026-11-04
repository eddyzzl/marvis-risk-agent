"""Formal risk-family dispatch authenticates and independently recomputes Vintage."""
import copy
import json
import shutil
from types import SimpleNamespace

import pytest

from marvis.orchestrator.eval.acceptance_trust import TrustInputError
from marvis.orchestrator.eval.formal_acceptance import _domain
from marvis.orchestrator.eval.formal_benchmark import FormalCase
from marvis.orchestrator.eval.runtime_contracts import digest
from marvis.orchestrator.eval.runtime_run_manifest import verify_run_manifest
from marvis.orchestrator.eval.runtime_scoring import ExpectedCase, ExpectedSuite
from test_acceptance_trust import put
from test_formal_benchmark import frozen_fixture
from test_runtime_archive_weighted_and_vintage import vintage_archive as vintage_archive


@pytest.fixture
def formal_vintage(vintage_archive, tmp_path):
    archive, sha, binding = vintage_archive
    shutil.copytree(archive, tmp_path / "archive")
    reference = put(tmp_path, "binding.json", binding.model_dump())
    public = archive.parents[2] / "public" / binding.run_id
    run = verify_run_manifest(public, expected_manifest_sha256=digest((public / "final-manifest.json").read_bytes()))
    expected = ExpectedSuite.model_validate_json((archive.parents[2] / "suite/private/expected.json").read_bytes()).cases[binding.case.id]
    template = next(case for case in frozen_fixture().cases if case.family == "risk_portfolio" and case.cohort == "normal")
    case = FormalCase.model_validate({**template.model_dump(), "id": binding.case.id,
        "case_sha256": digest(binding.case.model_dump()), "expected_sha256": digest(expected.model_dump())})
    return SimpleNamespace(case=case, expected=expected, binding=binding, root=tmp_path, run=run,
        item={"case_id": case.id, "archive": "archive", "manifest_sha256": sha, "binding": reference},
        record={"input_hashes": {reference["path"]: reference["sha256"]}})


def _run(f, expected=None):
    return _domain(f.case, expected or f.expected, f.item, f.root, f.record, f.run)


def test_formal_vintage_uses_retained_rows_and_keeps_acceptance_boundaries(formal_vintage):
    result, reason = _run(formal_vintage)
    assert reason is None
    assert result["domain_status"] == "verified" and result["actual_result"] == "done", result
    assert result["domain_checks"]["independent_vintage"]["status"] == "verified"
    assert dict(result["safety"])["authorization"] == "unavailable"


@pytest.mark.parametrize("damage", ["step", "material", "binding_hash", "definition"])
def test_formal_vintage_rejects_unbound_public_evidence(formal_vintage, damage):
    f = formal_vintage
    observed = f.run["executions"][f.case.id]
    if damage == "step":
        observed["execution"]["steps"][0]["output_sha256"] = "f" * 64
    elif damage == "material":
        observed["materials"][0]["sha256"] = "f" * 64
    elif damage == "binding_hash":
        f.record["input_hashes"].clear()
    else:
        f.case = f.case.model_copy(update={"case_sha256": "f" * 64})
    with pytest.raises(TrustInputError):
        _run(f)


def test_formal_vintage_compares_expectations_with_independent_values(formal_vintage):
    f = formal_vintage
    wrong = ExpectedCase(result="done", assertions=[{
        "kind": "output_length", "tool": "strategy.vintage_curve", "path": ["cohorts"], "value": 99,
    }])
    result, _ = _run(f, wrong)
    assert result["domain_status"] == "verified" and result["actual_result"] == "failed"
    unknown = copy.deepcopy(wrong.model_dump())
    unknown["assertions"][0]["path"] = ["business_approved"]
    with pytest.raises(TrustInputError, match="predicate_unavailable"):
        _run(f, ExpectedCase.model_validate(unknown))


def test_formal_vintage_does_not_promote_a_changed_curve(formal_vintage):
    f = formal_vintage
    # Even an archive-local rehash is rejected by its held external manifest.
    path = f.root / "archive/private.json"
    value = json.loads(path.read_text())
    value["passed"] = True
    path.write_text(json.dumps(value))
    result, _ = _run(f)
    assert result["domain_status"] == "mismatch" and result["actual_result"] == "failed"
